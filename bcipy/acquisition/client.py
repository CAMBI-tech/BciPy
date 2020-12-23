"""Data Acquisition Client"""
import logging
import multiprocessing
from multiprocessing import Queue
import time

from bcipy.acquisition import buffer_server
from bcipy.acquisition.record import Record
from bcipy.acquisition.util import StoppableProcess
from bcipy.acquisition.marker_writer import NullMarkerWriter, LslMarkerWriter
from bcipy.acquisition.connection_method import ConnectionMethod

log = logging.getLogger(__name__)
DEBUG = False
DEBUG_FREQ = 500
MSG_DEVICE_INFO = "device_info"
MSG_ERROR = "error"
MSG_PROCESSOR_INITIALIZED = "processor_initialized"


class CountClock():
    """Clock that provides timestamp values starting at 1.0; the next value
    is the increment of the previous. Implements the monotonic clock interface.
    """

    def __init__(self):
        super(CountClock, self).__init__()
        self.counter = 0

    def reset(self):
        """Resets the counter"""
        self.counter = 0

    # pylint: disable=invalid-name
    def getTime(self):
        """Gets the current count."""
        self.counter += 1
        return float(self.counter)


class DataAcquisitionClient:
    """Data Acquisition client. The client sets up a separate thread for
    acquisition, writes incoming data to a queue, and processes the data from
    the queue.

    Parameters
    ----------
        device: Device instance
            Object with device-specific implementations for connecting,
            initializing, and reading a packet.
        processor : Processor; optional
            A data Processor that does something with the streaming data (ex.
            writes to a file.)
        buffer_name : str, optional
            Name of the sql database archive; default is buffer.db.
        raw_data_file_name: str,
            Name of the raw data csv file to output; if not present raw data
                is not written.
        clock : Clock, optional
            Clock instance used to timestamp each acquisition record
        delete_archive: boolean, optional
            Flag indicating whether to delete the database archive on exit.
            Default is False.
    """

    def __init__(self,
                 device,
                 buffer_name='raw_data.db',
                 raw_data_file_name='raw_data.csv',
                 clock=CountClock(),
                 delete_archive=True):

        self._device = device
        self._buffer_name = buffer_name
        self._raw_data_file_name = raw_data_file_name
        self._clock = clock

        # boolean; set to false to retain the sqlite db.
        self.delete_archive = delete_archive

        self._device_info = None  # set on start.
        self._is_streaming = False

        # Offset in seconds from the start of acquisition to calibration
        # trigger. Calculated once, then cached.
        self._cached_offset = None
        self._record_at_calib = None
        self._max_wait = 0.1  # for process loop

        self.marker_writer = NullMarkerWriter()
        self._acq_process = None
        self._buf = None

    # @override ; context manager
    def __enter__(self):
        self.start_acquisition()
        return self

    # @override ; context manager
    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.stop_acquisition()

    def start_acquisition(self):
        """Run the initialization code and start the loop to acquire data from
        the server.

        We use multiprocessing to achieve best performance during our sessions.

        Some references:
            Stopping processes and other great multiprocessing examples:
                https://pymotw.com/2/multiprocessing/communication.html
            Windows vs. Unix Process Differences:
                https://docs.python.org/2.7/library/multiprocessing.html#windows

        """

        if not self._is_streaming:
            log.debug("Starting Acquisition")

            msg_queue = Queue()

            # Initialize the marker streams before the device connection so the
            # device can start listening.
            # TODO: ensure that connector is including marker streams
            if self._device.__class__.supports(self._device.device_spec, ConnectionMethod.LSL):
                self.marker_writer = LslMarkerWriter()

            # Clock is copied, so reset should happen in the main thread.
            self._clock.reset()

            # Used to communicate with the database from both the main thread
            # as well as the acquisition thread.
            self._buf = buffer_server.new_mailbox()

            self._acq_process = AcquisitionProcess(device=self._device,
                                                   clock=self._clock,
                                                   buf=self._buf,
                                                   msg_queue=msg_queue)
            self._acq_process.start()

            # Block thread until device connects and returns device_info.
            msg_type, msg = msg_queue.get()
            if msg_type == MSG_DEVICE_INFO:
                self._device_info = msg
                log.info("Connected to device")
                log.info(msg)
            elif msg_type == MSG_ERROR:
                raise Exception("Error connecting to device")
            else:
                raise Exception("Message not understood: " + str(msg))

            # Start up the database server
            buffer_server.start_server(self._buf, self._device_info.channels,
                                       self._buffer_name)
            # Inform acquisition process that database server is ready
            msg_queue.put(True)
            msg_queue = None
            self._is_streaming = True

    def stop_acquisition(self):
        """Stop acquiring data; perform cleanup."""
        log.debug("Stopping Acquisition Process")

        self._is_streaming = False

        self._acq_process.stop()
        self._acq_process.join()

        self.marker_writer.cleanup()
        self.marker_writer = NullMarkerWriter()

        if self._raw_data_file_name and self._buf:
            buffer_server.dump_data(self._buf, self._raw_data_file_name,
                                    self.device_info.name, self.device_info.fs)

    def get_data(self, start=None, end=None, field='_rowid_'):
        """Queries the buffer by field.

        Parameters
        ----------
            start : number, optional
                start of time slice; units are those of the acquisition clock.
            end : float, optional
                end of time slice; units are those of the acquisition clock.
            field: str, optional
                field on which to query; default value is the row id.
        Returns
        -------
            list of Records
        """

        if self._buf is None:
            return []

        return buffer_server.get_data(self._buf, start, end, field)

    def get_data_for_clock(self, calib_time: float, start_time: float,
                           end_time: float):
        """Queries the database, using start and end values relative to a
        clock different than the acquisition clock.

        Parameters
        ----------
            calib_time: float
                experiment_clock time (in seconds) at calibration.
            start_time : float, optional
                start of time slice; units are those of the experiment clock.
            end_time : float, optional
                end of time slice; units are those of the experiment clock.
        Returns
        -------
            list of Records
        """

        sample_rate = self._device_info.fs

        if self._record_at_calib is None:
            rownum_at_calib = 1
        else:
            rownum_at_calib = self._record_at_calib.rownum

        # Calculate number of samples since experiment_clock calibration;
        # multiplying by the fs converts from seconds to samples.
        start_offset = (start_time - calib_time) * sample_rate
        start = rownum_at_calib + start_offset

        end = None
        if end_time:
            end_offset = (end_time - calib_time) * sample_rate
            end = rownum_at_calib + end_offset

        return self.get_data(start=start, end=end, field='_rowid_')

    def get_data_len(self):
        """Efficient way to calculate the amount of data cached."""
        if self._buf is None:
            return 0
        return buffer_server.count(self._buf)

    @property
    def device_info(self):
        """Get the latest device_info."""
        return self._device_info

    @property
    def is_calibrated(self):
        """Returns boolean indicating whether or not acquisition has been
        calibrated (an offset calculated based on a trigger)."""
        return self.offset is not None

    @is_calibrated.setter
    def is_calibrated(self, bool_val):
        """Setter for the is_calibrated property that allows the user to
        override the calculated value and use a 0 offset.

        Parameters
        ----------
            bool_val: boolean
                if True, uses a 0 offset; if False forces the calculation.
        """
        self._cached_offset = 0.0 if bool_val else None

    @property
    def offset(self):
        """Offset in seconds from the start of acquisition to calibration
        trigger.

        Returns
        -------
            float or None if TRG channel is all 0.
        TODO: Consider setting the trigger channel name in the device_info.
        """

        # cached value if previously queried; only needs to be computed once.
        if self._cached_offset is not None:
            return self._cached_offset

        if self._buf is None or self._device_info is None:
            log.debug("Buffer or device has not been initialized")
            return None

        log.debug("Querying database for offset")
        # Assumes that the TRG column is present and used for calibration, and
        # that non-calibration values are all 0.
        rows = buffer_server.query(self._buf,
                                   filters=[("TRG", ">", 0)],
                                   ordering=("timestamp", "asc"),
                                   max_results=1)
        if not rows:
            log.debug("No rows have a TRG value.")
            return None

        log.debug(rows[0])
        # Calculate offset from the number of samples recorded by the time
        # of calibration.
        self._record_at_calib = rows[0]
        self._cached_offset = rows[0].rownum / self._device_info.fs
        log.debug("Cached offset: %s", str(self._cached_offset))
        return self._cached_offset

    @property
    def record_at_calib(self):
        """Data record at the calibration trigger"""
        return self._record_at_calib

    def cleanup(self):
        """Performs cleanup tasks, such as deleting the buffer archive. Note
        that data will be unavailable after calling this method."""
        if self._buf:
            buffer_server.stop(self._buf, delete_archive=self.delete_archive)
            self._buf = None


class AcquisitionProcess(StoppableProcess):
    """Process that continuously reads data from the data source and sends it
    to the buffer for persistence.


    Parameters
    ----------
        device: Device instance
            Object with device-specific implementations for connecting,
            initializing, and reading a packet.
        clock : Clock
            Used to timestamp each record as it is read.
        buf : Queue(s) used to send data to the database server
        msg_queue : Queue used to communicate with the main thread.
    """

    def __init__(self, device, clock, buf, msg_queue):
        super(AcquisitionProcess, self).__init__()
        self._device = device
        self._clock = clock
        self._buf = buf
        self.msg_queue = msg_queue

    def run(self):
        """Process startup. Connects to the device and start reading data.
        Since this is done in a separate process from the main thread, any
        errors encountered will be written to the msg_queue.
        """

        try:
            log.debug("Connecting to device")
            self._device.connect()
            self._device.acquisition_init()
        except Exception as error:
            self.msg_queue.put((MSG_ERROR, str(error)))
            raise error

        # Send updated device info to the main thread; this also signals that
        # initialization is complete.
        self.msg_queue.put((MSG_DEVICE_INFO, self._device.device_info))

        # Wait for db server start
        self.msg_queue.get()
        self.msg_queue = None

        log.debug("Starting Acquisition read data loop")
        sample = 0
        data = self._device.read_data()

        # begin continuous acquisition process as long as data received
        while self.running() and data:
            sample += 1
            if DEBUG and sample % DEBUG_FREQ == 0:
                log.debug("Read sample: %s", str(sample))

            buffer_server.append(self._buf, Record(data, self._clock.getTime(), sample))
            try:
                # Read data again
                data = self._device.read_data()
            # pylint: disable=broad-except
            except Exception as error:
                log.error("Error reading data from device: %s", str(error))
                data = None
                break
        log.debug("Total samples read: %s", str(sample))
        self._device.disconnect()


def main():
    """Test script."""
    import sys
    if sys.version_info >= (3, 0, 0):
        # Only available in Python 3; allows us to test process code as it
        # behaves in Windows environments.
        multiprocessing.set_start_method('spawn')

    import argparse
    import json
    from bcipy.acquisition.protocols import registry
    from bcipy.acquisition.devices import SUPPORTED_DEVICES
    from bcipy.acquisition.connection_method import ConnectionMethod

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--buffer', default='buffer.db',
                        help='buffer db name')
    parser.add_argument('-f', '--filename', default='rawdata.csv')
    parser.add_argument('-d', '--device', default='DSI',
                        choices=SUPPORTED_DEVICES.keys())
    parser.add_argument('-c', '--connection_method', default='LSL',
                        choices=ConnectionMethod.list())
    parser.add_argument('-p', '--params', type=json.loads,
                        default={'host': '127.0.0.1', 'port': 9000},
                        help="device connection params; json")
    args = parser.parse_args()

    device_builder = registry.find_device(
        args.device, ConnectionMethod.by_name(args.connection_method))

    # Instantiate and start collecting data
    dev = device_builder(connection_params=args.params)
    if args.channels:
        dev.channels = args.channels.split(',')
    daq = DataAcquisitionClient(device=dev,
                                buffer_name=args.buffer,
                                delete_archive=True)

    daq.start_acquisition()

    # Get data from buffer
    time.sleep(1)

    print("Number of samples in 1 second: {0}".format(daq.get_data_len()))

    time.sleep(1)

    print("Number of samples in 2 seconds: {0}".format(daq.get_data_len()))

    daq.stop_acquisition()
    daq.cleanup()


if __name__ == "__main__":
    main()
