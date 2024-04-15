from typing import List
from bcipy.task import TaskType
"""This file can define actions that can happen in a session orchestrator visit. To start these will be 1:1 with tasks, but later this can be extended to represent training sequences, GUI popups etc."""

taskname_dict = {}
for i, task in enumerate(TaskType.list()):
    taskname_dict[task] = TaskType(i + 1)

def parse_actions(action_sequence: str) -> List[TaskType]:
    """
    Parses a string of actions into a list of TaskType objects.

    Converts a string of actions into a list of TaskType objects. The string is expected to be in the format of 'Action1 -> Action2 -> ... -> ActionN'.

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
        actions = [taskname_dict[action.strip()] for action in action_sequence.split('->')]
    except KeyError as e:
        raise ValueError('Invalid task name in action sequence') from e
    return actions

def validate_action_string(action_sequence: str) -> None:
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
    try:
        for action in action_sequence.split('->'):
            if action.strip() not in taskname_dict:
                raise ValueError('Invalid task name in action sequence')
    except KeyError as e:
        raise ValueError('Invalid task name in action sequence') from e

def serialize_actions(action_sequence: List[TaskType]) -> str:
    """
    Converts a list of TaskType objects into a string of actions.

    Converts a list of TaskType objects into a string of actions. The string is in the format of 'Action1 -> Action2 -> ... -> ActionN'.

    Parameters
    ----------
        action_sequence : str
            A string of actions in the format of 'Action1 -> Action2 -> ... -> ActionN'.

    Returns
    -------
        List[TaskType]
            A list of TaskType objects that represent the actions in the input string.
    """
    action_str = ''
    for i, task in enumerate(action_sequence):
        action_str += task.label
        if i != len(action_sequence) - 1:
            action_str += ' -> '
    return action_str

if __name__ == '__main__':
    actions = parse_actions("Matrix Calibration -> Matrix Copy Phrase")
    string = serialize_actions(actions)
    print(actions)
    print(string)