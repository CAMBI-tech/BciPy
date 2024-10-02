"""
This import statement allows users to import submodules from Task
"""
from .main import Task, TaskData, TaskMode

# Makes the following classes available to the task registry
from .registry import TaskRegistry

__all__ = [
    'Task',
    'TaskRegistry',
    'TaskData',
    'TaskMode'
]
