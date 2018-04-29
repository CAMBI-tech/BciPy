from __future__ import absolute_import, division, print_function

import multiprocessing
import threading

class StoppableProcess(multiprocessing.Process):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the running() condition.

      https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread-in-python
    """

    def __init__(self, *args, **kwargs):
        super(StoppableProcess, self).__init__(*args, **kwargs)
        self._stopper = multiprocessing.Event()

    def stop(self):
        self._stopper.set()

    def running(self):
        return not self._stopper.is_set()

    def stopped(self):
        return self._stopper.is_set()


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the running() condition.
    """

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stopper = threading.Event()

    def stop(self):
        self._stopper.set()

    def running(self):
        return not self._stopper.isSet()

    def stopped(self):
        return self._stopper.isSet()