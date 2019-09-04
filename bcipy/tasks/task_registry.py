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
from itertools import groupby
from typing import Dict, List


class ExperimentType(Enum):
    """Enum of the registered experiment types (Tasks), along with the label
    used for display in the GUI and command line tools. Values are looked up
    by their (1-based) index.

    Examples:
    >>> ExperimentType(1)
    <ExperimentType.RSVP_CALIBRATION: 1>

    >>> ExperimentType(1).label
    'Calibration'
    """

    RSVP_CALIBRATION = 'Calibration'
    RSVP_COPY_PHRASE = 'Copy Phrase'
    RSVP_ICON_TO_ICON = 'Icon to Icon'
    RSVP_ICON_TO_WORD = 'Icon to Word'
    RSVP_ALERT_TONE_CALIBRATION = 'Alert Tone'
    RSVP_INTER_SEQUENCE_FEEDBACK_CALIBRATION = 'Feed. Calib.'
    RSVP_TIMING_VERIFICATION_CALIBRATION = 'Time Test'

    def __new__(cls, *args, **kwds):
        """Autoincrements the value of each item added to the enum."""
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, label):
        self.label = label

    @classmethod
    def by_mode(cls) -> Dict[str, List['ExperimentType']]:
        """ExperimentTypes by mode (ex. 'RSVP')"""

        def prefix(item, sep='_') -> str:
            name = item.name
            return name[0:name.index(sep)] if sep in name else ''

        mode_map = {}
        for key, group in groupby(cls, prefix):
            mode_map[key] = [task for task in group]
        return mode_map

    @classmethod
    def calibration_tasks(cls) -> List['ExperimentType']:
        return [task for task in cls
                if task.name.endswith('CALIBRATION') and 'COPY_PHRASE'
                not in task.name]
