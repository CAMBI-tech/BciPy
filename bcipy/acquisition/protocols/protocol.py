"""Defines the Protocol data structure"""
from collections import namedtuple

Protocol = namedtuple(
    'Protocol', ['encoder', 'init_messages', 'fs', 'channels'])
Protocol.__doc__ = ('Protocols are primarily used for generating data for '
                    'testing purposes')
