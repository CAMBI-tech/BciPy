# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import namedtuple

"""Domain object used for storing metadata about the collection parameters,
including device name, sample rate and the list of channels."""
DeviceInfo = namedtuple('DeviceInfo', ['fs', 'channels', 'name'])
