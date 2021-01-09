"""Tests for registry functions."""
import unittest

from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.devices import supported_device
from bcipy.acquisition.protocols.dsi.dsi_device import DsiDevice
from bcipy.acquisition.protocols.dsi.dsi_protocol import DsiProtocol
from bcipy.acquisition.protocols.lsl.lsl_device import LslDevice
from bcipy.acquisition.protocols.registry import find_device, find_protocol


class TestAcquisitionRegistry(unittest.TestCase):
    """Tests for acquisition registry module."""

    def test_find_protocol(self):
        """Registry should find the correct protocol given the DeviceSpec and ConnectionMethod"""
        device_spec = supported_device('DSI')
        protocol = find_protocol(device_spec, ConnectionMethod.TCP)
        self.assertTrue(isinstance(protocol, DsiProtocol))

    def test_find_connector(self):
        """Registry should find the correct connector given the DeviceSpec and ConnectionMethod"""
        self.assertEqual(
            find_device(supported_device('DSI'), ConnectionMethod.TCP),
            DsiDevice)
        self.assertEqual(
            find_device(supported_device('DSI'), ConnectionMethod.LSL),
            LslDevice)
        self.assertEqual(
            find_device(supported_device('LSL'), ConnectionMethod.LSL),
            LslDevice)
