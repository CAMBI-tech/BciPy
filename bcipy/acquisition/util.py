
import multiprocessing
import threading


class StoppableProcess(multiprocessing.Process):
    """Process class with a stop() method. The process itself has to check
    regularly for the running() condition.
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
        return not self._stopper.is_set()

    def stopped(self):
        return self._stopper.is_set()