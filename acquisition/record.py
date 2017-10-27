from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import namedtuple

"""Domain object used for storing data and timestamp information, where data
is a single reading from a device and is a list of channel information
(float)."""
Record = namedtuple('Record', ['data', 'timestamp'])
