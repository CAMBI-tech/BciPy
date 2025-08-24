# -*- coding: utf-8 -*-
"""Defines the Record namedtuple"""
from typing import Any, List, NamedTuple, Optional


class Record(NamedTuple):
    """Domain object used for storing data and timestamp information.

    The `data` attribute represents a single reading from a device and is a list
    of channel information (typically float values).

    Attributes:
        data (List[Any]): A list of values representing channel information from a device.
        timestamp (float): The timestamp associated with the data recording.
        rownum (Optional[int], optional): The row number of the record, if applicable.
                                          Defaults to None.
    """
    data: List[Any]
    timestamp: float
    rownum: Optional[int] = None
