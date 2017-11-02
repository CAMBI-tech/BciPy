"""
Binary protocol for DSI; ported to construct 2.8 from the original
rsvpkeyboard app.
"""

from __future__ import absolute_import, division, print_function

# from __future__ import (unicode_literals)
from construct import (Array, Bytes, Const, Embedded, Enum, Float32b, Int8ub,
                       Int16ub, Int32ub, Optional, PascalString, Struct,
                       Switch, this)

header = Struct(
    'magic' / Const(b'@ABCD'),                           # bytes 0-5
    'type' / Enum(Int8ub, NULL=0, EEG_DATA=1, EVENT=5),  # byte 5
    'payload_length' / Int16ub,                          # bytes 6-7
    'number' / Int32ub,                                  # bytes 8-11
)

header_len = 12

# TODO: Resolve the 'unknown encoding' error for the message when we import
# unicode_literals.
event = Struct(
    'event_code' / Enum(Int32ub,  # bytes 12-15
                        VERSION=1,
                        DATA_START=2,
                        DATA_STOP=3,
                        SENSOR_MAP=9,
                        DATA_RATE=10
                        ),
    'sending_node' / Int32ub,  # bytes 16-19

    # Message data is optional
    # Previous implementation used If(this._.payload_length > 8, ...), but this
    # method does not seem to work in v2.8.

    # message_length: bytes 20-23, message: bytes 24+
    'message' / Optional(PascalString(lengthfield='message_length' / Int32ub,
                                      encoding='ascii'))
)

EEG_data = Struct(
    'timestamp' / Float32b,   # bytes 12-15
    'data_counter' / Int8ub,  # byte 16; Unused, just 0 currently
    'ADC_status' / Bytes(6),  # bytes 17-22
    # bytes 23-26, 27-30, etc.
    'sensor_data' / Array((this._.payload_length - 11) // 4, Float32b)
)

null = Struct('none' / Array(111, Int8ub))

packet = Struct(
    Embedded(header),
    'payload' / Embedded(Switch(this.type,
                                {
                                    'NULL': null,
                                    'EVENT': event,
                                    'EEG_DATA': EEG_data
                                }))
)
