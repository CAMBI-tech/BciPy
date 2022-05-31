import logging

from abc import ABC, abstractmethod


class Task(ABC):
    """Task.

    Base class for BciPy tasks.
    """

    def __init__(self):
        super(Task, self).__init__()
        self.logger = logging.getLogger(__name__)

    def configure(self):
        ...

    @abstractmethod
    def execute(self):
        ...

    @abstractmethod
    def name(self):
        ...
