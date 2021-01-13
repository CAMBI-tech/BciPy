"""Acquisition connection methods."""

from enum import Enum, auto
from typing import List


class ConnectionMethod(Enum):
    """Supported methods for connecting to an acquisition device. Each device_spec
    must specify which connection methods are appropriate."""
    TCP = auto()
    LSL = auto() # LabStreamingLayer
    # Other options may include USB, etc.

    @classmethod
    def by_name(cls, name):
        """Returns the ConnectionMethod with the given name."""
        items = cls.list()
        # The cls.list method returns a sorted list of enum tasks
        # check if item present and return the index + 1 (which is the ENUM value for the task)
        if name in items:
            return cls(items.index(name) + 1)

    @classmethod
    def list(cls) -> List[str]:
        """Returns the list of ConnectionMethod names"""
        return list(map(lambda c: c.name, cls))