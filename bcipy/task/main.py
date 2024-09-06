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
    save_path: Optional[str] = None
    task_dict: Optional[dict] = None


class Task(ABC):
    """Task.

    Base class for BciPy tasks.
    """
    name: str
    parameters: Parameters
    data_save_location: str
    logger: logging.Logger

    def __init__(self, *args, **kwargs) -> None:
        super(Task, self).__init__()
        assert getattr(self, 'name', None) is not None, "Task must have a `name` attribute defined"

    @abstractmethod
    def execute(self) -> TaskData:
        ...

    def setup(self, parameters, data_save_location):
        self.parameters = parameters
        self.data_save_location = data_save_location
        ...

    def cleanup(self):
        ...
