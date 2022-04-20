"""Task Registry ; used to provide task options to the GUI and command line
tools. User defined tasks can be added to the Registry."""


# NOTE:
# In the future we may want to consider dynamically retrieving all subclasses
# of Task and use these to populate a registry. We could also provide
# functionality for bcipy users to define their own tasks and register them so
# they would appear as options in the GUI.
#
# However, this approach is currently problematic for the GUI interface. Due
# to the tight coupling of the display code with the Tasks, importing any of
# the Task subclasses causes pygame (a psychopy dependency) to create a GUI,
# which seems to prevent our other GUI code from working.

from enum import Enum
from typing import List

from bcipy.helpers.exceptions import BciPyCoreException


class TaskType(Enum):
    """Enum of the registered experiment types (Tasks), along with the label
    used for display in the GUI and command line tools. Values are looked up
    by their (1-based) index.

    Examples:
    >>> TaskType(1)
    <TaskType.RSVP_CALIBRATION: 1>

    >>> TaskType(1).label
    'RSVP Calibration'
    """

    RSVP_CALIBRATION = 'RSVP Calibration'
    RSVP_COPY_PHRASE = 'RSVP Copy Phrase'
    RSVP_TIMING_VERIFICATION_CALIBRATION = 'RSVP Time Test Calibration'
    MATRIX_CALIBRATION = 'Matrix Calibration'
    MATRIX_TIMING_VERIFICATION_CALIBRATION = 'Matrix Time Test Calibration'

    def __new__(cls, *args, **kwds):
        """Autoincrements the value of each item added to the enum."""
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, label):
        self.label = label

    @classmethod
    def by_value(cls, item):
        tasks = cls.list()
        # The cls.list method returns a sorted list of enum tasks
        # check if item present and return the index + 1 (which is the ENUM value for the task)
        if item in tasks:
            return cls(tasks.index(item) + 1)
        raise BciPyCoreException(f'{item} not a registered TaskType={tasks}')

    @classmethod
    def calibration_tasks(cls) -> List['TaskType']:
        return [task for task in cls
                if task.name.endswith('CALIBRATION') and 'COPY_PHRASE'
                not in task.name]

    @classmethod
    def list(cls) -> List[str]:
        return list(map(lambda c: c.label, cls))
