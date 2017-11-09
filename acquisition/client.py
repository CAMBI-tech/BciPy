"""Data Acquisition Client"""
from __future__ import absolute_import, division, print_function

import logging
import Queue
import threading
import time
import timeit

from buffer import Buffer
from processor import FileWriter
from record import Record

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
    """

    def __init__(self,
                 device,
                 processor=FileWriter.builder('rawdata.csv'),
                 buffer=Buffer.builder('buffer.db'),
                 clock=_Clock()):

        self._device = device
        self._make_processor = processor
        self._make_buffer = buffer
        self._clock = clock

        self._is_streaming = False

        self._initial_wait = 5  # for process loop
        multiplier = self._device.fs if self._device.fs else 100
        maxsize = (self._initial_wait + 1) * multiplier
        self._process_queue = Queue.Queue(maxsize=maxsize)

    # @override ; context manager
    def __enter__(self):
        self.start_acquisition()
        return self

    # @override ; context manager
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_acquisition()

    def start_acquisition(self):
        """Run the initialization code and start the loop to acquire data from
        the server."""

        self._device.connect()

        if not self._is_streaming:
            logging.debug("Starting acquisition")
            self._is_streaming = True

            # Read headers/params
            self._device.acquisition_init()

            # Initialize the buffer and processor; this occurs after the
            # device initialization to ensure that any device parameters have
            # been updated as needed.
            self._buf = self._make_buffer(channels=self._device.channels)
            self._processor = self._make_processor(self._device.name,
                                                   self._device.fs,
                                                   self._device.channels)

            self._acq_thread = _StoppableThread(target=self._acquisition_loop)
            self._process_thread = _StoppableThread(target=self._process_loop)
            self._process_thread.daemon = True
            self._process_thread.start()
            self._acq_thread.start()

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
                    # decrease the wait after data has been initially received
                    wait = 1
                except Queue.Empty:
                    break
                self._buf.append(record)
                p.process(record.data, record.timestamp)

    def _acquisition_loop(self):
        """Continuously reads data from the source and sends it to the buffer
        for processing."""

        if self._is_streaming:
            data = self._device.read_data()
            while self._acq_thread.running() and data:
                self._process_queue.put(Record(data, self._clock.getTime()))
                data = self._device.read_data()

    def stop_acquisition(self):
        """Stop acquiring data; perform cleanup."""

        self._is_streaming = False

        self._device.disconnect()
        self._acq_thread.stop()
        self._acq_thread.join()

        # allow initial_wait seconds to wrap up any queued work
        counter = 0
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


class _StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the running() condition.

      https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread-in-python
    """

    def __init__(self, *args, **kwargs):
        super(_StoppableThread, self).__init__(*args, **kwargs)
        self._stopper = threading.Event()

    def stop(self):
        self._stopper.set()

    def running(self):
        return not self._stopper.isSet()

    def stopped(self):
        return self._stopper.isSet()


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
                        default={'host': '127.0.0.1', 'port': 8844},
                        help="device connection params; json")
    args = parser.parse_args()

    Device = registry.find_device(args.device)

    # Instantiate and start collecting data
    channels = args.channels.split(',') if args.channels else []
    daq = Client(device=Device(connection_params=args.params,
                               channels=channels),
                 processor=FileWriter.builder(args.filename),
                 buffer=Buffer.builder(args.buffer))

    daq.start_acquisition()

    # Get data from buffer
    time.sleep(1)

    print("Number of samples in 1 second: {0}".format(daq.get_data_len()))

    time.sleep(1)

    print("Number of samples in 2 seconds: {0}".format(daq.get_data_len()))

    daq.stop_acquisition()
