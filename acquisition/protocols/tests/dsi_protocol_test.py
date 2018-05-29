
import protocols.dsi.dsi as dsi
import protocols.dsi.dsi_protocol as dsi_protocol
import unittest

class TestDsiProtocol(unittest.TestCase):
    """Tests for DsiProtocol"""


    def test_protocol_init_messages(self):
        """Should have the channels and the sample_rate."""
        p = dsi_protocol.DsiProtocol()
        channel_msg = p.init_messages[0]
        c = dsi.packet.parse(channel_msg)
        self.assertEqual(c.type, 'EVENT')
        self.assertEqual(c.event_code, 'SENSOR_MAP')
        self.assertEqual(c.message, ','.join(dsi.DEFAULT_CHANNELS))

        c2 = dsi.packet.parse(p.init_messages[1])
        self.assertEqual(c2.type, 'EVENT')
        self.assertEqual(c2.event_code, 'DATA_RATE')
        self.assertEqual(c2.message, u',300')


    def test_encoder(self):
        """It should encode array data that can be subsequently decoded."""

        data = [float(i) for i in range(25)]
        encoder = dsi_protocol.Encoder()
        encoded = encoder.encode(data)
        c = dsi.packet.parse(encoded)
        self.assertEqual(c.type, 'EEG_DATA')
        self.assertEqual(c.sensor_data, data)
