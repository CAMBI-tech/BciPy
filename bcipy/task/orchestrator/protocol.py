"""This file can define actions that can happen in a session orchestrator visit.
To start these will be 1:1 with tasks, but later this can be extended to represent training sequences, GUI popups etc"""

from typing import List, Type

from bcipy.config import TASK_SEPERATOR
from bcipy.task import Task
from bcipy.task.registry import TaskRegistry


def parse_protocol(protocol: str) -> List[Type[Task]]:
    """
    Parses a string of actions into a list of Task objects.

    Converts a string of actions into a list of Task objects. The string is expected
    to be in the format of 'Action1 -> Action2 -> ... -> ActionN'.
    Parameters
    ----------
        protocol : str
            A string of actions in the format of 'Action1 -> Action2 -> ... -> ActionN'.

    Returns
    -------
        List[TaskType]
            A list of TaskType objects that represent the actions in the input string.
    """
    task_registry = TaskRegistry()
    return [task_registry.get(item.strip()) for item in protocol.split(TASK_SEPERATOR)]


def validate_protocol_string(protocol: str) -> None:
    """
    Validates a string of actions.

    Validates a string of actions. The string is expected to be in the format of 'Action1 -> Action2 -> ... -> ActionN'.

    Parameters
    ----------
        protocol : str
            A string of actions in the format of 'Action1 -> Action2 -> ... -> ActionN'.

    Raises
    ------
        ValueError
            If the string of actions is invalid.
    """
    for protocol_item in protocol.split(TASK_SEPERATOR):
        if protocol_item.strip() not in TaskRegistry().list():
            raise ValueError(f"Invalid task '{protocol_item}' name in protocol string.")


def serialize_protocol(protocol: List[Type[Task]]) -> str:
    """
    Converts a list of TaskType objects into a string of actions.

    Converts a list of TaskType objects into a string of actions. The string is in the format of
    'Action1 -> Action2 -> ... -> ActionN'.

    Parameters
    ----------
        protocol : str
            A string of actions in the format of 'Action1 -> Action2 -> ... -> ActionN'.

    Returns
    -------
        List[TaskType]
            A list of TaskType objects that represent the actions in the input string.
    """

    return f" {TASK_SEPERATOR} ".join([item.name for item in protocol])


if __name__ == '__main__':
    actions = parse_protocol("Matrix Calibration -> Matrix Copy Phrase")
    string = serialize_protocol(actions)
    print(actions)
    print(string)
