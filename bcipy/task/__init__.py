"""
This import statement allows users to import submodules from Task
"""
from .main import Task
from .task_registry import TaskType

# Makes the following classes available to the task registry
import bcipy.orchestrator.actions
from . import base_calibration
from .paradigm import rsvp, vep, matrix
from .task_registry import TaskRegistry

__all__ = [
    'Task',
    'TaskType',
    'TaskRegistry'
]
