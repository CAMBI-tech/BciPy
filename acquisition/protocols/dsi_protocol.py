"""Protocol with information for generating and serving mock DSI data."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import timeit

import acquisition.protocols.dsi as dsi
import acquisition.protocols.protocol as protocol


default_fs = 300
default_channels = ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz',
                    'CM', 'A1', 'Fp1', 'Fp2', 'T3', 'T5', 'O1', 'O2',
                    'X3', 'X2', 'F7', 'F8', 'X1',
                    'A2', 'T6', 'T4', 'TRG']


def DsiProtocol(fs=default_fs, channels=default_channels):
    """Protocol for mocking DSI data."""
    return protocol.Protocol(encoder=Encoder(),
                             init_messages=_init_messages(fs, channels),
                             fs=fs,
                             channels=channels)


def _event_packet(code, msg):
    """Construct an event packet with the given event code and message."""

    # calculate payload length
    event_code_bytes, sending_node_bytes, msg_len_bytes, msg_bytes = (
        4, 4, 4, len(msg))

    payload_len = sum(
        [event_code_bytes, sending_node_bytes, msg_len_bytes, msg_bytes])

    params = dict(type='EVENT', payload_length=payload_len, number=13,
                  event_code=code, sending_node=33,
                  message_length=len(msg), message=msg)

    return dsi.packet.build(params)


def _init_messages(fs, channels):
    """Messages sent at the start of the initialization protocol when
    connecting with a client. Sent before any data is sent.

    Parameters
    ----------
        fs : int
            sample frequency
        channels : list
            list of channel names
    """

    s_msg = ','.join(channels).encode('ascii', 'ignore')
    sensor_msg = _event_packet('SENSOR_MAP', s_msg)

    # Comma-delimited list; the frequency is the second param, which is the
    # only one we use; not sure what else is in this list.
    f_msg = (',' + str(fs)).encode('ascii', 'ignore')
    freq_msg = _event_packet('DATA_RATE', f_msg)

    return [sensor_msg, freq_msg]


class Encoder(object):
    """Encodes sensor data as binary data in the DSI format."""

    def __init__(self):
        super(Encoder, self).__init__()
        self.start_time = timeit.default_timer()

    def encode(self, sensor_data):
        """Builds a binary data packet from the provided sensor data. Timestamp
        is the difference from the given start_time.

        Parameters
        ----------
            sensor_data: list - list of sensor values (float type); len must
                match the channel_count

        Returns
        -------
        Binary data for a single packet.
        """
        # (timestamp/4 bytes, counter/1 byte, status/6 bytes) + 4 bytes for
        # every sensor/float.
        payload_length = (11 + (len(sensor_data) * 4))

        params = dict(type='EEG_DATA',
                      payload_length=payload_length,
                      number=13,  # arbitrary numberr
                      timestamp=(timeit.default_timer() - self.start_time),
                      data_counter=0,  # unused
                      ADC_status=b"123456",  # unused
                      sensor_data=sensor_data)

        return dsi.packet.build(params)
