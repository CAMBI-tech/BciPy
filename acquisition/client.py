"""Data Acquisition Client"""
from __future__ import absolute_import, division, print_function

import logging
import multiprocessing
import threading
import time
import timeit

from queue import Empty


from buffer import Buffer
from processor import FileWriter
from record import Record
from util import StoppableThread, StoppableProcess

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


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
        processor : function -> Processor; optional
            Constructor for a Processor (contextmanager with a `process`
            method)
        buffer : function -> Buffer, optional
            Constructor for a Buffer
        clock : Clock, optional
            Clock instance used to timestamp each acquisition record
        offset: integer, used to help calibrate timing between device and the apps that use it.
    """

    def __init__(self,
                 device,
                 processor=FileWriter(filename='rawdata.csv'),
                 buffer_name='buffer.db',
                 clock=_Clock()):

        self._device = device
        self._processor = processor
        self._buffer_name = buffer_name
        self._clock = clock

        self._is_streaming = False
        self._is_calibrated = False
        # offset in seconds from the start of acquisition to calibration trigger
        self.offset = 0

        self._initial_wait = 2  # for process loop
        multiplier = self._device.fs if self._device.fs else 100
        maxsize = (self._initial_wait + 1) * multiplier

        self._process_queue = multiprocessing.Queue(maxsize=maxsize)

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

        We use threading and multiprocessing to achieve best performance during
        our sessions.

        ****
        Eventually, we'd like to parallelize both acquisition and processing
        loop but there are some issues with how our process loop is designed
        and cannot work on Windows. This is due to os forking vs. duplication.
            There are fixes in #Python3, but we are currently tied to v2.7
                -- unix systems can directly replace StoppableThread with
                    StoppableProcess!
        ****

        Some references:
            Stopping processes and other great multiprocessing examples:
                https://pymotw.com/2/multiprocessing/communication.html
            Windows vs. Unix Process Differences:
                https://docs.python.org/2.7/library/multiprocessing.html#windows

        """

        if not self._is_streaming:
            logging.debug("Starting Acquisition")

            self._is_streaming = True

            self._acq_process = AcquisitionProcess(self._device, self._clock,
                                                   self._process_queue)
            self._acq_process.start()

            # Initialize the buffer and processor; this occurs after the
            # device initialization to ensure that any device parameters have
            # been updated as needed.

            # TODO: Now that device initialization has been moved to the
            # acquisition loop, and that loop is in its own process,
            # self._device is copied and never updated. So the device_info set
            # in the processor and buffer only reflects what was initially
            # configured. Initialization should either be moved back into this
            # method or we need to use inter-process communication to update
            # these values.

            self._buf = Buffer(channels=self._device.channels,
                               archive_name=self._buffer_name)
            self._processor.set_device_info(self._device.name, self._device.fs,
                                            self._device.channels)

            self._process_thread = StoppableThread(target=self._process_loop)
            self._process_thread.daemon = True
            self._process_thread.start()

    def _process_loop(self):
        """Reads from the queue of data and performs processing an item at a
        time. Also writes data to buffer. Intended to be in its own thread."""

        assert self._process_thread.running()

        with self._processor as p:
            wait = self._initial_wait
            while self._process_thread.running():
                try:
                    # block if necessary
                    record = self._process_queue.get(True, wait)

                    # if device not calibrated, look for the first trigger
                    # signal as a marker of starting location.
                    if not self._is_calibrated:
                        # This assumes the last record is the trigger channel!
                        if record.data[-1] > 0:
                            self._is_calibrated = True

                            # TODO: device.fs may not be correct. See above.
                            self.offset = record.timestamp / self._device.fs

                    # decrease the wait after data has been initially received
                    wait = 2
                except Empty:
                    break
                self._buf.append(record)
                p.process(record.data, record.timestamp)

    def stop_acquisition(self):
        """Stop acquiring data; perform cleanup."""
        logging.debug("Stopping Acquisition Process")

        self._is_streaming = False

        self._acq_process.stop()
        self._acq_process.join()

        # allow initial_wait seconds to wrap up any queued work
        counter = 0
        logging.debug("Stopping Processing Queue")
        while not self._process_queue.empty() and \
                counter < (self._initial_wait * 100):
            counter += 1
            time.sleep(0.1)

        self._process_thread.stop()
        self._process_thread.join()
        self._buf.close()

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
        elif start is None:
            return self._buf.all()
        else:
            return self._buf.query(start, end)

    def get_data_len(self):
        """Efficient way to calculate the amount of data cached."""
        if self._buf is None:
            return 0
        else:
            return len(self._buf)

    def cleanup(self):
        """Performs cleanup tasks, such as deleting the buffer archive. Note
        that data may be unavailable after calling this method."""
        if self._buf:
            self._buf.cleanup()


class AcquisitionProcess(StoppableProcess):
    def __init__(self, device, clock, data_queue):
        super(AcquisitionProcess, self).__init__()
        self._device = device
        self._clock = clock  # TODO: clock manager?
        self._data_queue = data_queue

    def run(self):
        """Continuously reads data from the source and sends it to the buffer
        for processing.
        """
        # TODO: Send updated device information (fs, channels).
        self._device.connect()
        self._device.acquisition_init()
        self._clock.reset()

        sample = 1
        data = self._device.read_data()

        # begin continuous acquisition process as long as data received
        while self.running() and data:
            # Use get time to timestamp and continue saving records.
            self._data_queue.put(
                Record(data, sample))
            try:
                # Read data again
                data = device.read_data()
                sample += 1
            except:
                data = None
                break
        self._device.disconnect()


if __name__ == "__main__":

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
                 buffer_name=args.buffer)

    daq.start_acquisition()

    # Get data from buffer
    time.sleep(1)

    print("Number of samples in 1 second: {0}".format(daq.get_data_len()))

    time.sleep(1)

    print("Number of samples in 2 seconds: {0}".format(daq.get_data_len()))

    daq.stop_acquisition()
