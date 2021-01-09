"""Tests for the DSI headset Protocol."""
import unittest
from bcipy.acquisition.protocols.dsi import dsi
from bcipy.acquisition.protocols.dsi.dsi_protocol import DsiProtocol
from bcipy.acquisition.devices import DeviceSpec


class TestDsiProtocol(unittest.TestCase):
    """Tests for DsiProtocol"""

    def test_protocol_init_messages(self):
        """Should have the channels and the sample_rate."""
        device_spec = DeviceSpec(
            name="DSI-VR300",
            channels=["P4", "Fz", "Pz", "F7", "PO8", "PO7", "Oz", "TRG"],
            sample_rate=300.0)
        protocol = DsiProtocol(device_spec)
        init_messages = protocol.init_messages()
        channel_msg = init_messages[0]
        parsed1 = dsi.packet.parse(channel_msg)
        self.assertEqual(parsed1.type, 'EVENT')
        self.assertEqual(parsed1.event_code, 'SENSOR_MAP')
        self.assertEqual(parsed1.message, ','.join(device_spec.channels))

        parsed2 = dsi.packet.parse(init_messages[1])
        self.assertEqual(parsed2.type, 'EVENT')
        self.assertEqual(parsed2.event_code, 'DATA_RATE')
        self.assertEqual(parsed2.message, u',300')

    def test_encoder(self):
        """It should encode array data that can be subsequently decoded."""
        device_spec = DeviceSpec(
            name="DSI-VR300",
            channels=["P4", "Fz", "Pz", "F7", "PO8", "PO7", "Oz", "TRG"],
            sample_rate=300.0)
        protocol = DsiProtocol(device_spec)
        data = [float(i) for i in range(device_spec.channel_count)]
        encoded = protocol.encode(data)
        parsed = dsi.packet.parse(encoded)
        self.assertEqual(parsed.type, 'EEG_DATA')
        self.assertEqual(parsed.sensor_data, data)
