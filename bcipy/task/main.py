import logging
from bcipy.helpers.parameters import Parameters
from abc import ABC, abstractmethod


class Task(ABC):
    """Task.

    Base class for BciPy tasks.
    """

    parameters: Parameters
    data_save_location: str

    def __init__(self) -> None:
        super(Task, self).__init__()
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def execute(self) -> str:
        assert self.parameters is not None, "Task parameters not set"
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def setup(self, parameters, data_save_location):
        self.parameters = parameters
        self.data_save_location = data_save_location
        ...

    def cleanup(self):
        ...
