"""Data Acquisition Client"""
from __future__ import absolute_import, division, print_function

import logging
import multiprocessing
import time
import timeit

from queue import Empty

import buffer_server
# from buffer import Buffer
from processor import FileWriter
from record import Record
from util import StoppableThread, StoppableProcess

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
DEBUG = False
DEBUG_FREQ = 500


class _Clock(object):
    """Default clock that uses the timeit module to generate timestamps"""

    def __init__(self):
        super(_Clock, self).__init__()
        self.reset()

    def reset(self):
        self._reset_at = timeit.default_timer()

    def getTime(self):
        return timeit.default_timer() - self._reset_at


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
        self.delete_archive = delete_archive

        self._is_streaming = False

        # offset in seconds from the start of acquisition to calibration
        # trigger
        self._cached_offset = None
        self._initial_wait = 0.1  # for process loop

        # Max number of records in queue before it blocks for processing.
        maxsize = 100

        self._process_queue = multiprocessing.JoinableQueue(maxsize=maxsize)

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

            acq_started = multiprocessing.Event()

            # Device connection must happen in the Main thread if using
            # the lsl_device due to limitations in the underlying lib.
            self._device.connect()
            self._device.acquisition_init()
            self._clock.reset()

            self._acq_process = AcquisitionProcess(self._device, self._clock,
                                                   self._process_queue,
                                                   acq_started)
            self._acq_process.start()

            # Initialize the buffer and processor; this occurs after the
            # device initialization to ensure that any device parameters have
            # been updated as needed.

            # Set/update device info. This may have been set in the device
            # during the acquisition_init.
            self._processor.set_device_info(self._device.name, self._device.fs,
                                            self._device.channels)
            self._buf = buffer_server.start(channels=self._device.channels,
                                            archive_name=self._buffer_name)

            self._data_processor = DataProcessor(data_queue=self._process_queue,
                                                 processor=self._processor,
                                                 buf=self._buf,
                                                 wait=self._initial_wait)

            acq_started.wait()
            self._data_processor.start()
            self._is_streaming = True

    def stop_acquisition(self):
        """Stop acquiring data; perform cleanup."""
        logging.debug("Stopping Acquisition Process")

        self._is_streaming = False

        self._acq_process.stop()
        self._acq_process.join()
        self._device.disconnect()

        logging.debug("Stopping Processing Queue")

        # Blocks until all data in the queue is consumed.
        self._process_queue.join()
        self._data_processor.stop()

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
    def is_calibrated(self):
        return self.offset != None

    @property
    def offset(self):
        """Offset in seconds from the start of acquisition to calibration
        trigger.

        Returns
        -------
            float or None if TRG channel is all 0.
        """

        # cached value if previously queried; only needs to be computed once.
        if self._cached_offset:
            return self._cached_offset

        if self._buf is None or self._device is None:
            logging.debug("Buffer or device has not been initialized")
            return None

        rows = buffer_server.query(self._buf,
                                   filters=[("TRG", ">", 0)],
                                   ordering=("timestamp", "asc"),
                                   max_results=1)
        if len(rows) == 0:
            return None
        else:
            self._cached_offset = rows[0].timestamp / self._device.fs
            return self._cached_offset

    def cleanup(self):
        """Performs cleanup tasks, such as deleting the buffer archive. Note
        that data will be unavailable after calling this method."""
        if self._buf:
            buffer_server.stop(self._buf, delete_archive=self.delete_archive)
            self._buf = None


class AcquisitionProcess(StoppableProcess):
    def __init__(self, device, clock, data_queue, start_event):
        super(AcquisitionProcess, self).__init__()
        self._device = device
        self._clock = clock
        self._data_queue = data_queue
        self._start_event = start_event

    def run(self):
        """Continuously reads data from the source and sends it to the buffer
        for processing.
        """

        logging.debug("Starting Acquisition read data loop")
        self._start_event.set()
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
