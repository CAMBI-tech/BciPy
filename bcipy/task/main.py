import logging
from dataclasses import dataclass
from typing import Optional
from bcipy.helpers.parameters import Parameters
from abc import ABC, abstractmethod

from bcipy.helpers.stimuli import play_sound
from bcipy.config import STATIC_AUDIO_PATH


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

    def setup(self, *args, **kwargs):
        ...

    def cleanup(self, *args, **kwargs):
        ...

    def alert(self):
        play_sound(f"{STATIC_AUDIO_PATH}/{self.parameters['alert_sound_file']}")
