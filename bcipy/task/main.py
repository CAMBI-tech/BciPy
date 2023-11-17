import logging

from abc import ABC, abstractmethod


class Task(ABC):
    """Task.

    Base class for BciPy tasks.
    """

    def __init__(self) -> None:
        super(Task, self).__init__()
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def execute(self) -> str:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    def configure(self) -> None:
        ...
