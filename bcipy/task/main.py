"""Core task module defining base classes for BciPy tasks.

This module provides the foundational classes for implementing BCI tasks,
including the abstract base Task class and supporting data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from bcipy.config import STATIC_AUDIO_PATH
from bcipy.core.parameters import Parameters
from bcipy.core.stimuli import play_sound


@dataclass
class TaskData:
    """Data structure for storing task execution results.

    This class encapsulates the data returned from a task execution, including
    the save path for any generated data and a dictionary of task-specific data.

    Attributes:
        save_path: Path where task data was saved.
        task_dict: Dictionary containing task-specific data and results.
    """
    save_path: Optional[str] = None
    task_dict: Optional[Dict[str, Any]] = None


class TaskMode(Enum):
    """Enumeration of supported BCI task modes.

    This enum defines the different types of tasks that can be executed in the BCI system.
    Each mode represents a specific type of interaction or experiment.

    Attributes:
        CALIBRATION: Mode for system calibration tasks.
        COPYPHRASE: Mode for copy-spelling tasks.
        TIMING_VERIFICATION: Mode for timing verification tasks.
        ACTION: Mode for action-based tasks.
        TRAINING: Mode for training tasks.
    """
    CALIBRATION = "calibration"
    COPYPHRASE = "copy phrase"
    TIMING_VERIFICATION = "timing verification"
    ACTION = "action"
    TRAINING = "training"

    def __str__(self) -> str:
        """Return the string value of the task mode.

        Returns:
            str: The string representation of the task mode.
        """
        return self.value

    def __repr__(self) -> str:
        """Return the string representation of the task mode.

        Returns:
            str: The string representation of the task mode.
        """
        return self.value


class Task(ABC):
    """Abstract base class for BciPy tasks.

    This class defines the interface that all BCI tasks must implement. It provides
    the basic structure for task execution, setup, and cleanup.

    Attributes:
        name: Name of the task.
        mode: Mode of operation for the task.
        parameters: Task configuration parameters.
        data_save_location: Location where task data should be saved.

    Note:
        Subclasses must define the 'name' and 'mode' class attributes and
        implement the execute() method.
    """
    name: str
    mode: TaskMode
    parameters: Parameters
    data_save_location: str

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the task.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            AssertionError: If name or mode attributes are not defined.
        """
        super(Task, self).__init__()
        assert getattr(
            self, 'name', None) is not None, "Task must have a `name` attribute defined"
        assert getattr(
            self, 'mode', None) is not None, "Task must have a `mode` attribute defined"

    @abstractmethod
    def execute(self) -> TaskData:
        """Execute the task.

        This method must be implemented by all task subclasses to define the
        task's execution logic.

        Returns:
            TaskData: Object containing the results of the task execution.
        """
        ...

    def setup(self, *args, **kwargs) -> None:
        """Set up the task before execution.

        This method can be overridden by subclasses to perform any necessary
        setup before task execution.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        ...

    def cleanup(self, *args, **kwargs) -> None:
        """Clean up after task execution.

        This method can be overridden by subclasses to perform any necessary
        cleanup after task execution.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        ...

    def alert(self) -> None:
        """Play an alert sound.

        Plays the configured alert sound file to notify the user.
        The sound file is specified in the task parameters.
        """
        play_sound(
            f"{STATIC_AUDIO_PATH}/{self.parameters['alert_sound_file']}")
