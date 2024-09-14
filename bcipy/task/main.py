import logging
from dataclasses import dataclass
from typing import Optional
from bcipy.helpers.parameters import Parameters
from abc import ABC, abstractmethod
from enum import Enum


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

    class Mode(str, Enum):
        COPYPHRASE = 'COPYPHRASE'
        CALIBRATION = 'CALIBRATION'
        ACTION = 'ACTION'

    # Inherits from str so that it is automatically json serializable
    class Paradigm(str, Enum):
        """Paradigm.

        Enum for task paradigms.
        """
        RSVP = 'RSVP'
        MATRIX = 'MATRIX'
        VEP = 'VEP'

    name: str
    parameters: Parameters
    data_save_location: str
    logger: logging.Logger
    mode: 'Task.Mode'

    def is_calibration(self) -> bool:
        return self.mode == Task.Mode.CALIBRATION

    def is_copyphrase(self) -> bool:
        return self.mode == Task.Mode.COPYPHRASE

    def is_action(self) -> bool:
        return self.mode == Task.Mode.ACTION

    def __init__(self, *args, **kwargs) -> None:
        super(Task, self).__init__()
        assert getattr(self, 'name', None) is not None, "Task must have a `name` attribute defined"
        assert getattr(self, 'mode', None) is not None, "Task must have a valid `mode` attribute defined"

    @abstractmethod
    def execute(self) -> TaskData:
        ...

    def setup(self, parameters, data_save_location):
        self.parameters = parameters
        self.data_save_location = data_save_location
        ...

    def cleanup(self):
        ...
