from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import construct
import protocols.dsi.dsi as dsi
import pytest
import unittest

# Expected values were generated from existing construct v.2.5.5 structs in the
# existing rsvpkeyboard daq module.


class TestDsi(unittest.TestCase):

    def test_header(self):
        expected = b'@ABCD\x00\x00\x80\x00\x00\x00\x0c'
        params = dict(type='NULL', payload_length=128, number=12)
        result = dsi.header.build(params)
        self.assertEquals(result, expected)

        c = dsi.header.parse(expected)
        for k in params.keys():
            self.assertEqual(c[k], params[k])

    def test_none_packet(self):

        expected = b'@ABCD\x00\x00\x10\x00\x00\x00\r\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmn'  # noqa
        params = dict(type='NULL', payload_length=16, number=13,
                      none=[i for i in range(111)])
        result = dsi.packet.build(params)
        self.assertEqual(result, expected)

        c = dsi.packet.parse(expected)
        for k in params.keys():
            self.assertEqual(c[k], params[k])

    def test_event_packet(self):
        expected = b'@ABCD\x05\x00\x08\x00\x00\x00\r\x00\x00\x00\x02\x00\x00\x00!\x00\x00\x00\x0bhello world'  # noqa
        params = dict(type='EVENT', payload_length=8, number=13,
                      event_code='DATA_START', sending_node=33,
                      message_length=11, message='hello world')

        result = dsi.packet.build(params)
        self.assertEqual(result, expected)

        c = dsi.packet.parse(expected)
        for k in params.keys():
            if k != 'message_length':
                self.assertEqual(c[k], params[k])

    def test_event_sensor_map(self):
        channels = u'P3,C3,F3,Fz,F4,C4,P4,Cz,CM,A1,Fp1,Fp2,T3,T5,O1,O2,X3,X2,F7,F8,X1,A2,T6,T4,TRG'.encode(  # noqa
            'ascii', 'ignore')
        event_code_bytes, sending_node_bytes, msg_len_bytes, msg_bytes = (
            4, 4, 4, len(channels) * 4)
        payload_length = sum([event_code_bytes, sending_node_bytes,
                              msg_len_bytes, msg_bytes])
        params = dict(type='EVENT', payload_length=payload_length, number=13,
                      event_code='SENSOR_MAP', sending_node=33,
                      message_length=len(channels), message=channels)

        expected = b'@ABCD\x05\x01@\x00\x00\x00\r\x00\x00\x00\t\x00\x00\x00!\x00\x00\x00MP3,C3,F3,Fz,F4,C4,P4,Cz,CM,A1,Fp1,Fp2,T3,T5,O1,O2,X3,X2,F7,F8,X1,A2,T6,T4,TRG'  # noqa

        result = dsi.packet.build(params)
        self.assertEqual(result, expected)

        c = dsi.packet.parse(expected)
        channel_names = c.message.split(",")
        self.assertEqual(len(channel_names), 25)

    def test_eeg_packet(self):
        expected = b'@ABCD\x01\x00\x13\x00\x00\x00\rN\x05Y\xe9\x00123456C\xf2ff\xc3\xfb\x19\x9a'  # noqa

        params = dict(type='EEG_DATA', payload_length=19, number=13,
                      timestamp=559315520.0, data_counter=0,
                      ADC_status=b"123456", sensor_data=[484.799987793,
                                                         -502.200012207])
        result = dsi.packet.build(params)
        self.assertEqual(result, expected)

        c = dsi.packet.parse(expected)
        for k in params.keys():
            if k != 'sensor_data':
                self.assertEqual(c[k], params[k])
        self.assertEqual(("%.9f" % c['sensor_data'][0]), '484.799987793')
        self.assertEqual(("%.9f" % c['sensor_data'][1]), '-502.200012207')

        # Payload size that exceeds sensor_data points should throw an error
        with pytest.raises(construct.core.RangeError):
            params['payload_length'] = 24
            dsi.packet.build(params)
