"""This file can define actions that can happen in a session orchestrator visit.
To start these will be 1:1 with tasks, but later this can be extended to represent training sequences, GUI popups etc"""

from typing import List
from bcipy.task import Task, TaskType
from bcipy.orchestrator.actions import task_registry_dict

ACTION_SEPARATOR = '->'

# task_registry_dict = {}
# for i, task in enumerate(TaskType.list()):
#     assert task not in task_registry_dict
#     task_registry_dict[task] = TaskType(i + 1)

# for action in Action.get_all_actions():
#     assert action.label not in task_registry_dict, f"Conflicting definitions for action {action.label}"
#     task_registry_dict[action.label] = action


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
        sequence = [task_registry_dict[action.strip()] for action in sequence.split(ACTION_SEPARATOR)]
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
    for sequence_item in action_sequence.split(ACTION_SEPARATOR):
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
    return f" {ACTION_SEPARATOR} ".join([item.label for item in sequence])


if __name__ == '__main__':
    actions = parse_sequence("Matrix Calibration -> Matrix Copy Phrase")
    string = serialize_sequence(actions)
    print(actions)
    print(string)
