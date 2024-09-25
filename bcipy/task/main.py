from dataclasses import dataclass
from typing import Optional
from bcipy.helpers.parameters import Parameters
from abc import ABC, abstractmethod
from enum import Enum

from bcipy.helpers.stimuli import play_sound
from bcipy.config import STATIC_AUDIO_PATH


@dataclass
class TaskData():
    """TaskData.

    Data structure for storing task return data.
    """
    save_path: Optional[str] = None
    task_dict: Optional[dict] = None

class TaskMode(Enum):
    CALIBRATION = "calibration"
    COPYPHRASE = "copy phrase"
    TIMING_VERIFICATION = "timing verification"
    ACTION = "action"
    TRAINING = "training"

    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return self.value


class Task(ABC):
    """Task.

    Base class for BciPy tasks.
    """
    name: str
    mode: TaskMode
    parameters: Parameters
    data_save_location: str

    def __init__(self, *args, **kwargs) -> None:
        super(Task, self).__init__()
        assert getattr(self, 'name', None) is not None, "Task must have a `name` attribute defined"
        assert getattr(self, 'mode', None) is not None, "Task must have a `mode` attribute defined"

    @abstractmethod
    def execute(self) -> TaskData:
        ...

    def setup(self, *args, **kwargs):
        ...

    def cleanup(self, *args, **kwargs):
        ...

    def alert(self):
        play_sound(f"{STATIC_AUDIO_PATH}/{self.parameters['alert_sound_file']}")
