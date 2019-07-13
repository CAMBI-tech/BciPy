"""Tests for the DSI headset Protocol."""
import unittest
from bcipy.acquisition.protocols.dsi import dsi
from bcipy.acquisition.protocols.dsi import dsi_protocol


class TestDsiProtocol(unittest.TestCase):
    """Tests for DsiProtocol"""

    def test_protocol_init_messages(self):
        """Should have the channels and the sample_rate."""
        protocol = dsi_protocol.DsiProtocol()
        channel_msg = protocol.init_messages[0]
        parsed1 = dsi.packet.parse(channel_msg)
        self.assertEqual(parsed1.type, 'EVENT')
        self.assertEqual(parsed1.event_code, 'SENSOR_MAP')
        self.assertEqual(parsed1.message, ','.join(dsi.DEFAULT_CHANNELS))

        parsed2 = dsi.packet.parse(protocol.init_messages[1])
        self.assertEqual(parsed2.type, 'EVENT')
        self.assertEqual(parsed2.event_code, 'DATA_RATE')
        self.assertEqual(parsed2.message, u',300')

    def test_encoder(self):
        """It should encode array data that can be subsequently decoded."""

        data = [float(i) for i in range(25)]
        encoder = dsi_protocol.Encoder()
        encoded = encoder.encode(data)
        parsed = dsi.packet.parse(encoded)
        self.assertEqual(parsed.type, 'EEG_DATA')
        self.assertEqual(parsed.sensor_data, data)
