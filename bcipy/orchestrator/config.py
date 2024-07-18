"""This file can define actions that can happen in a session orchestrator visit.
To start these will be 1:1 with tasks, but later this can be extended to represent training sequences, GUI popups etc"""

from typing import List, Type
from bcipy.task import Task
from bcipy.config import TASK_SEPERATOR
from bcipy.task.task_registry import TaskRegistry


def parse_sequence(sequence: str) -> List[Type[Task]]:
    """
    Parses a string of actions into a list of TaskType objects.

    Converts a string of actions into a list of TaskType objects. The string is expected
    to be in the format of 'Action1 -> Action2 -> ... -> ActionN'.
    Parameters
    ----------
        action_sequence : str
            A string of actions in the format of 'Action1 -> Action2 -> ... -> ActionN'.

    Returns
    -------
        List[TaskType]
            A list of TaskType objects that represent the actions in the input string.
    """
    task_registry = TaskRegistry()
    return [task_registry.get(item.strip()) for item in sequence.split(TASK_SEPERATOR)]


def validate_sequence_string(action_sequence: str) -> None:
    """
    Validates a string of actions.

    Validates a string of actions. The string is expected to be in the format of 'Action1 -> Action2 -> ... -> ActionN'.

    Parameters
    ----------
        action_sequence : str
            A string of actions in the format of 'Action1 -> Action2 -> ... -> ActionN'.

    Raises
    ------
        ValueError
            If the string of actions is invalid.
    """
    for sequence_item in action_sequence.split(TASK_SEPERATOR):
        if sequence_item.strip() not in TaskRegistry().list():
            raise ValueError(f"Invalid task '{sequence_item}' name in action sequence")


def serialize_sequence(sequence: List[Type[Task]]) -> str:
    """
    Converts a list of TaskType objects into a string of actions.

    Converts a list of TaskType objects into a string of actions. The string is in the format of
    'Action1 -> Action2 -> ... -> ActionN'.

    Parameters
    ----------
        action_sequence : str
            A string of actions in the format of 'Action1 -> Action2 -> ... -> ActionN'.

    Returns
    -------
        List[TaskType]
            A list of TaskType objects that represent the actions in the input string.
    """

    return f" {TASK_SEPERATOR} ".join([item.name for item in sequence])


if __name__ == '__main__':
    actions = parse_sequence("Matrix Calibration -> Matrix Copy Phrase")
    string = serialize_sequence(actions)
    print(actions)
    print(string)
