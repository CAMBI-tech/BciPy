"""Data Acquisition Client"""
import logging
import multiprocessing
import time

from queue import Empty

import buffer_server
from processor import FileWriter
from record import Record
from util import StoppableProcess
from marker_writer import NullMarkerWriter, LslMarkerWriter

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
DEBUG = False
DEBUG_FREQ = 500
MSG_DEVICE_INFO = "device_info"
MSG_ERROR = "error"


class _Clock(object):
    """Clock that provides timestamp values starting at 1.0; the next value
    is the increment of the previous."""

    def __init__(self):
        super(_Clock, self).__init__()
        self.counter = 0

    def reset(self):
        self.counter = 0

    def getTime(self):
        self.counter += 1
        return float(self.counter)


class Client(object):
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
        clock : Clock, optional
            Clock instance used to timestamp each acquisition record
        delete_archive: boolean, optional
            Flag indicating whether to delete the database archive on exit.
            Default is True.
    """

    def __init__(self,
                 device,
                 processor=FileWriter(filename='rawdata.csv'),
                 buffer_name='buffer.db',
                 clock=_Clock(),
                 delete_archive=True):

        self._device = device
        self._processor = processor
        self._buffer_name = buffer_name
        self._clock = clock

        # boolean; set to false to retain the sqlite db.
        self.delete_archive = delete_archive

        self._device_info = None  # set on start.
        self._is_streaming = False

        # Offset in seconds from the start of acquisition to calibration
        # trigger. Calculated once, then cached.
        self._cached_offset = None
        self._max_wait = 0.1  # for process loop

        # Max number of records in queue before it blocks for processing.
        maxsize = 500

        self._process_queue = multiprocessing.JoinableQueue(maxsize=maxsize)
        self.marker_writer = NullMarkerWriter()

    # @override ; context manager
    def __enter__(self):
        self.start_acquisition()
        return self

    # @override ; context manager
    def __exit__(self, exc_type, exc_value, traceback):
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
            logging.debug("Starting Acquisition")

            msg_queue = multiprocessing.Queue()

            # Initialize the marker streams before the device connection so the
            # device can start listening.
            # TODO: Should this be a property of the device?
            if self._device.name == 'LSL':
                self.marker_writer = LslMarkerWriter()

            # Clock is copied, so reset should happen in the main thread.
            self._clock.reset()

            self._acq_process = AcquisitionProcess(self._device, self._clock,
                                                   self._process_queue,
                                                   msg_queue)
            self._acq_process.start()

            # Block thread until device connects and returns device_info.
            msg_type, msg = msg_queue.get()
            if msg_type == MSG_DEVICE_INFO:
                self._device_info = msg
            elif msg_type == MSG_ERROR:
                raise Exception("Error connecting to device")
            else:
                raise Exception("Message not understood: " + str(msg))

            # Initialize the buffer and processor; this occurs after the
            # device initialization to ensure that any device parameters have
            # been updated as needed.
            self._processor.set_device_info(self._device_info)
            self._buf = buffer_server.start(self._device_info.channels,
                                            self._buffer_name)

            self._data_processor = DataProcessor(data_queue=self._process_queue,
                                                 processor=self._processor,
                                                 buf=self._buf,
                                                 wait=self._max_wait)
            self._data_processor.start()
            self._is_streaming = True

    def stop_acquisition(self):
        """Stop acquiring data; perform cleanup."""
        logging.debug("Stopping Acquisition Process")

        self._is_streaming = False

        self._acq_process.stop()
        self._acq_process.join()

        logging.debug("Stopping Processing Queue")

        # Blocks until all data in the queue is consumed.
        self._process_queue.join()
        self._data_processor.stop()
        self.marker_writer.cleanup()
        self.marker_writer = NullMarkerWriter()

    def get_data(self, start=None, end=None):
        """ Gets data from the buffer.

        Parameters
        ----------
            start : float, optional
                start of time slice
            end : float, optional
                end of time slice

        Returns
        -------
            list of Records
        """

        if self._buf is None:
            return []
        else:
            return buffer_server.get_data(self._buf, start, end)

    def get_data_len(self):
        """Efficient way to calculate the amount of data cached."""
        if self._buf is None:
            return 0
        else:
            return buffer_server.count(self._buf)

    @property
    def device_info(self):
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
            logging.debug("Buffer or device has not been initialized")
            return None

        logging.debug("Querying database for offset")
        rows = buffer_server.query(self._buf,
                                   filters=[("TRG", ">", 0)],
                                   ordering=("timestamp", "asc"),
                                   max_results=1)
        if len(rows) == 0:
            logging.debug("No rows have a TRG value.")
            return None
        else:
            self._cached_offset = rows[0].timestamp / self._device_info.fs
            logging.debug("Cached offset: " + str(self._cached_offset))
            return self._cached_offset

    def cleanup(self):
        """Performs cleanup tasks, such as deleting the buffer archive. Note
        that data will be unavailable after calling this method."""
        if self._buf:
            buffer_server.stop(self._buf, delete_archive=self.delete_archive)
            self._buf = None


class AcquisitionProcess(StoppableProcess):
    def __init__(self, device, clock, data_queue, msg_queue):
        super(AcquisitionProcess, self).__init__()
        self._device = device
        self._clock = clock
        self._data_queue = data_queue
        self._msg_queue = msg_queue

    def run(self):
        """Continuously reads data from the source and sends it to the buffer
        for processing.
        """

        try:
            logging.debug("Connecting to device")
            self._device.connect()
            self._device.acquisition_init()
        except Exception as e:
            self._msg_queue.put((MSG_ERROR, str(e)))
            raise e

        # Send updated device info to the main thread; this also signals that
        # initialization is complete.
        self._msg_queue.put((MSG_DEVICE_INFO, self._device.device_info))

        logging.debug("Starting Acquisition read data loop")
        sample = 0
        data = self._device.read_data()

        # begin continuous acquisition process as long as data received
        while self.running() and data:
            sample += 1
            if DEBUG and sample % DEBUG_FREQ == 0:
                logging.debug("Read sample: " + str(sample))

            self._data_queue.put(Record(data, self._clock.getTime()))
            try:
                # Read data again
                data = self._device.read_data()
            except Exception as e:
                logging.debug("Error reading data from device: " + str(e))
                data = None
                break
        logging.debug("Total samples read: " + str(sample))
        self._device.disconnect()


class DataProcessor(StoppableProcess):
    """Process that gets data from a queue and persists it to the buffer."""

    def __init__(self, data_queue, processor, buf, wait=0.1):
        super(DataProcessor, self).__init__()
        self._data_queue = data_queue
        self._processor = processor
        self._buf = buf
        self._wait = wait

    def run(self):
        """Reads from the queue of data and performs processing an item at a
        time. Also writes data to buffer."""

        logging.debug("Starting Data Processing loop.")
        count = 0
        with self._processor as p:
            while self.running():
                try:
                    record = self._data_queue.get(True, self._wait)
                    count += 1
                    if DEBUG and count % DEBUG_FREQ == 0:
                        logging.debug("Processed sample: " + str(count))
                    buffer_server.append(self._buf, record)
                    p.process(record.data, record.timestamp)
                    self._data_queue.task_done()
                except Empty:
                    pass
            logging.debug("Total samples processed: " + str(count))


if __name__ == "__main__":

    import sys
    if sys.version_info >= (3, 0, 0):
        # Only available in Python 3; allows us to test process code as it
        # behaves in Windows environments.
        multiprocessing.set_start_method('spawn')

    import argparse
    import json
    import protocols.registry as registry

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--buffer', default='buffer.db',
                        help='buffer db name')
    parser.add_argument('-f', '--filename', default='rawdata.csv')
    parser.add_argument('-d', '--device', default='DSI',
                        choices=registry.supported_devices.keys())
    parser.add_argument('-c', '--channels', default='',
                        help='comma-delimited list')
    parser.add_argument('-p', '--params', type=json.loads,
                        default={'host': '127.0.0.1', 'port': 9000},
                        help="device connection params; json")
    args = parser.parse_args()

    Device = registry.find_device(args.device)

    # Instantiate and start collecting data
    device = Device(connection_params=args.params)
    if args.channels:
        device.channels = args.channels.split(',')
    daq = Client(device=device,
                 processor=FileWriter(filename=args.filename),
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
