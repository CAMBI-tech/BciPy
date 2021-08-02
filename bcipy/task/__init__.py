"""
This import statement allows users to import submodules from Task
"""
from .main import Task
from .task_registry import TaskType

__all__ = [
    'Task',
    'TaskType',
]
