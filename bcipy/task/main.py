import logging
from dataclasses import dataclass
from typing import Optional
from bcipy.helpers.parameters import Parameters
from abc import ABC, abstractmethod


@dataclass
class TaskData():
    """TaskData.

    Data structure for storing task return data.
    """
    task_save: Optional[str] = None


class Task(ABC):
    """Task.

    Base class for BciPy tasks.
    """
    name: str
    parameters: Parameters
    data_save_location: str

    def __init__(self) -> None:
        super(Task, self).__init__()
        assert getattr(self, 'name', None) is not None, "Task must have a `name` attribute defined"
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def execute(self) -> TaskData:
        ...

    def setup(self, parameters, data_save_location):
        self.parameters = parameters
        self.data_save_location = data_save_location
        ...

    def cleanup(self):
        ...
