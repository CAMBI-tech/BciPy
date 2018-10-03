# -*- coding: utf-8 -*-
"""Defines the Record namedtuple"""
from collections import namedtuple

Record = namedtuple('Record', ['data', 'timestamp', 'rownum'])
Record.__doc__ = """Domain object used for storing data and timestamp
information, where data is a single reading from a device and is a list
of channel information (float)."""
