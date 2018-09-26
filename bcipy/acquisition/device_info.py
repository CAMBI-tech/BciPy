# -*- coding: utf-8 -*-
"""Defines the DeviceInfo namedtuple."""
from collections import namedtuple

DeviceInfo = namedtuple('DeviceInfo', ['fs', 'channels', 'name'])
DeviceInfo.__doc__ = """Domain object used for storing metadata about the
collection parameters, including device name, sample rate and the
list of channels."""
