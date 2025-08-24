"""Protocol handling module for BciPy task orchestration.

This module provides functionality for parsing and managing task protocols,
which define sequences of actions to be executed in a session. While currently
focused on task sequences, this can be extended to support training sequences,
GUI interactions, and other orchestrated behaviors.
"""

from typing import List, Type

from bcipy.config import TASK_SEPARATOR
from bcipy.task import Task
from bcipy.task.registry import TaskRegistry


def parse_protocol(protocol: str) -> List[Type[Task]]:
    """Parse a protocol string into a list of Task classes.

    Converts a string of task names into a list of Task classes. The string
    should be in the format 'Task1 -> Task2 -> ... -> TaskN', where each
    task name corresponds to a registered task in the TaskRegistry.

    Args:
        protocol: String of task names separated by the task separator.
            Format: 'Task1 -> Task2 -> ... -> TaskN'

    Returns:
        List[Type[Task]]: List of Task classes corresponding to the protocol.

    Raises:
        ValueError: If any task name in the protocol is not registered.
    """
    task_registry = TaskRegistry()
    return [task_registry.get(item.strip()) for item in protocol.split(TASK_SEPARATOR)]


def validate_protocol_string(protocol: str) -> None:
    """Validate a protocol string against registered tasks.

    Checks that all task names in the protocol string correspond to
    registered tasks in the TaskRegistry.

    Args:
        protocol: String of task names separated by the task separator.
            Format: 'Task1 -> Task2 -> ... -> TaskN'

    Raises:
        ValueError: If any task name in the protocol is not registered.
    """
    for protocol_item in protocol.split(TASK_SEPARATOR):
        if protocol_item.strip() not in TaskRegistry().list():
            raise ValueError(
                f"Invalid task '{protocol_item}' name in protocol string.")


def serialize_protocol(tasks: List[Type[Task]]) -> str:
    """Convert a list of Task classes into a protocol string.

    Creates a protocol string from a list of Task classes, using the task
    separator to join task names.

    Args:
        tasks: List of Task classes to serialize.

    Returns:
        str: Protocol string in format 'Task1 -> Task2 -> ... -> TaskN'.
    """
    return f" {TASK_SEPARATOR} ".join([item.name for item in tasks])
