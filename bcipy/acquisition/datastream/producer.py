"""Code for mocking an EEG data stream. Code in this module produces data
at a specified frequency."""
import logging
import random
import threading
import time
from builtins import next
from queue import Queue
from typing import Any, Iterator, Optional

from bcipy.acquisition.datastream.generator import random_data_generator
from bcipy.config import SESSION_LOG_FILENAME

log = logging.getLogger(SESSION_LOG_FILENAME)


class Producer(threading.Thread):
    """Produces generated data at a specified frequency.

    This class extends `threading.Thread` to run data generation in a separate
    thread, pushing generated samples into a queue at a specified frequency.
    It can also act as a context manager to automatically start and stop the
    thread.

    Args:
        queue (Queue): The queue where generated data will be written.
        freq (float, optional): The frequency (in Hz) at which data will be generated.
                                Defaults to 1/100 (0.01 Hz).
        generator (Optional[Any], optional): A Python generator for creating data.
                                             Defaults to `random_data_generator()`.
        maxiters (Optional[int], optional): If provided, stops generating data
                                            after this many iterations. Defaults to None.
    """

    def __init__(self,
                 queue: Queue,
                 freq: float = 1 / 100,
                 generator: Optional[Any] = None,
                 maxiters: Optional[int] = None):

        super(Producer, self).__init__()
        self.daemon: bool = True
        self._running: bool = True

        self.freq: float = freq
        self.generator: Any = generator or random_data_generator()
        self.maxiters: Optional[int] = maxiters
        self.queue: Queue = queue

    def __enter__(self) -> 'Producer':
        """Enters the runtime context related to this object.

        Starts the producer thread when entering the context.

        Returns:
            Producer: The Producer instance.
        """
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        """Exits the runtime context related to this object.

        Stops the producer thread when exiting the context.

        Args:
            _exc_type (Any): The exception type, if an exception was raised.
            _exc_value (Any): The exception value, if an exception was raised.
            _traceback (Any): The traceback, if an exception was raised.
        """
        self.stop()

    def _genitem(self) -> Any:
        """Generates the data item to be added to the queue.

        Returns:
            Any: The next data item from the configured generator.
        """
        return next(self.generator)

    def _additem(self) -> None:
        """Adds the data item to the queue.

        This method fetches an item using `_genitem` and places it into the internal queue.
        """
        self.queue.put(self._genitem())

    def stop(self) -> None:
        """Stops the thread.

        Sets an internal flag to stop the thread's execution and waits for the thread to finish.
        Stopped threads cannot be restarted.
        """

        self._running = False
        self.join()

    def run(self) -> None:
        """Provides a control loop, adding a data item to the queue at the
        configured frequency.

        This method overrides the `threading.Thread.run` method.
        """

        def tick() -> Iterator[float]:
            """Corrects the time interval if the time of the work to add the
            item causes drift.

            Yields:
                float: The calculated sleep duration to maintain the target frequency.
            """
            current_time: float = time.time()
            count: int = 0
            while True:
                count += 1
                yield max(current_time + count * self.freq - time.time(), 0)

        sleep_len: Iterator[float] = tick()
        times: int = 0
        while self._running and (self.maxiters is None or
                                 times < self.maxiters):
            times += 1
            time.sleep(next(sleep_len))
            self._additem()


class _ConsumerThread(threading.Thread):
    """Consumer used to test the Producer by consuming generated items.

    Args:
        queue (Queue): The queue from which to consume items.
        name (Optional[str], optional): The name of the consumer thread.
                                        Defaults to None.
    """

    def __init__(self, queue: Queue, name: Optional[str] = None):
        super(_ConsumerThread, self).__init__(name=name)
        self.daemon: bool = True
        self._q: Queue = queue

    def run(self) -> None:
        """Main loop for the consumer thread.

        Continuously checks the queue for items and processes them, logging
        the item and the queue size.
        """
        while True:
            if not self._q.empty():
                item: Any = self._q.get()
                log.info('Getting %s: %s items in queue',
                         str(item), str(self._q.qsize()))
                time.sleep(random.random())


def main() -> None:
    """Main method to demonstrate the Producer and Consumer threads.

    Initializes a Producer and a Consumer thread, starts them, and lets them run
    for a short period (5 seconds) before the program exits.
    """
    data_queue: Queue = Queue()
    producer: Producer = Producer(data_queue)
    consumer: _ConsumerThread = _ConsumerThread(data_queue)

    producer.start()
    consumer.start()
    time.sleep(5)


if __name__ == '__main__':
    main()
