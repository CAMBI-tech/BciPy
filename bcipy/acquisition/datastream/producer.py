"""Code for mocking an EEG data stream. Code in this module produces data
at a specified frequency."""
import logging
from builtins import next
from queue import Queue
import random
import threading
import time

from bcipy.acquisition.datastream.generator import random_data_generator

log = logging.getLogger(__name__)


class Producer(threading.Thread):
    """Produces generated data at a specified frequency.

    Parameters
    ----------
        queue : Queue
            Generated data will be written to the queue.
        freq : float, optional
            Data will be generated at the given frequency.
        generator : object, optional
            python generator for creating data.
        maxiters : int, optional
            if provided, stops generating data after the given number of iters.
    """

    def __init__(self,
                 queue,
                 freq=1 / 100,
                 generator=random_data_generator(),
                 maxiters=None):

        super(Producer, self).__init__()
        self.daemon = True
        self._running = True

        self.freq = freq
        self.generator = generator
        self.maxiters = maxiters
        self.queue = queue

    # @override to make this class a context manager
    def __enter__(self):
        self.start()
        return self

    # @override to make this class a context manager
    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.stop()

    def _genitem(self):
        """Generates the data item to be added to the queue."""

        try:
            data = next(self.generator)
        except StopIteration:
            log.debug("End of input reached")
            raise Exception("End of input reached")
        return data

    def _additem(self):
        """Adds the data item to the queue."""

        self.queue.put(self._genitem())

    def stop(self):
        """Stop the thread; stopped threads cannot be restarted."""

        self._running = False

    def run(self):
        """Provides a control loop, adding a data item to the queue at the
        configured frequency.

        @overrides the Thread run method
        """

        def tick():
            """Corrects the time interval if the time of the work to add the
            item causes drift."""
            current_time = time.time()
            count = 0
            while True:
                count += 1
                yield max(current_time + count * self.freq - time.time(), 0)

        sleep_len = tick()
        times = 0
        while self._running and (self.maxiters is None or
                                 times < self.maxiters):
            times += 1
            time.sleep(next(sleep_len))
            self._additem()


class _ConsumerThread(threading.Thread):
    """Consumer used to test the Producer by consuming generated items."""

    def __init__(self, queue, name=None):
        super(_ConsumerThread, self).__init__()
        self.daemon = True
        self.name = name
        self._q = queue

    def run(self):
        while True:
            if not self._q.empty():
                item = self._q.get()
                log.debug('Getting %s: %s items in queue',
                          str(item), str(self._q.qsize()))
                time.sleep(random.random())


def main():
    """Main method"""
    data_queue = Queue()
    producer = Producer(data_queue)
    consumer = _ConsumerThread(data_queue)

    producer.start()
    consumer.start()
    time.sleep(5)


if __name__ == '__main__':
    main()
