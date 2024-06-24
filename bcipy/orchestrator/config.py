"""This file can define actions that can happen in a session orchestrator visit.
To start these will be 1:1 with tasks, but later this can be extended to represent training sequences, GUI popups etc"""

from typing import List
from bcipy.task import Task
from bcipy.orchestrator.actions import task_registry_dict
from bcipy.config import TASK_SEPERATOR


def parse_sequence(sequence: str) -> List[Task]:
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
    try:
        sequence = [task_registry_dict[task.strip()] for task in sequence.split(TASK_SEPERATOR)]
    except KeyError as e:
        raise ValueError('Invalid task name in action sequence') from e
    return sequence


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
        if sequence_item.strip() not in task_registry_dict:
            raise ValueError('Invalid task name in action sequence')


def serialize_sequence(sequence: List[Task]) -> str:
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
    return f" {TASK_SEPERATOR} ".join([item.label for item in sequence])


if __name__ == '__main__':
    actions = parse_sequence("Matrix Calibration -> Matrix Copy Phrase")
    string = serialize_sequence(actions)
    print(actions)
    print(string)
