# -*- coding: utf-8 -*-
"""Defines the Record namedtuple"""
from typing import Any, List, NamedTuple


class Record(NamedTuple):
    """Domain object used for storing data and timestamp
    information, where data is a single reading from a device and is a list
    of channel information (float)."""
    data: List[Any]
    timestamp: float
    rownum: int = None
