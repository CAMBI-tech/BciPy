"""Tests for registry functions."""
import unittest

from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.devices import preconfigured_device
from bcipy.acquisition.protocols.dsi.dsi_connector import DsiConnector
from bcipy.acquisition.protocols.dsi.dsi_protocol import DsiProtocol
from bcipy.acquisition.protocols.lsl.lsl_connector import LslConnector
from bcipy.acquisition.protocols.registry import find_connector, find_protocol, make_connector


class TestAcquisitionRegistry(unittest.TestCase):
    """Tests for acquisition registry module."""

    def test_find_protocol(self):
        """Registry should find the correct protocol given the DeviceSpec and ConnectionMethod"""
        device_spec = preconfigured_device('DSI')
        protocol = find_protocol(device_spec, ConnectionMethod.TCP)
        self.assertTrue(isinstance(protocol, DsiProtocol))

    def test_find_connector(self):
        """Registry should find the correct connector given the DeviceSpec and ConnectionMethod"""
        self.assertEqual(
            find_connector(preconfigured_device('DSI'), ConnectionMethod.TCP),
            DsiConnector)
        self.assertEqual(
            find_connector(preconfigured_device('DSI'), ConnectionMethod.LSL),
            LslConnector)
        self.assertEqual(
            find_connector(preconfigured_device('LSL'), ConnectionMethod.LSL),
            LslConnector)

    def test_connector(self):
        """Registry should construct the correct connector given the DeviceSpec and ConnectionMethod."""
        connector = make_connector(preconfigured_device('DSI'),
                                   ConnectionMethod.TCP, {
                                       'host': '127.0.0.1',
                                       'port': 9000
        })
        self.assertTrue(isinstance(connector, DsiConnector))
        self.assertEqual(9000, connector.connection_params['port'])

        connector = make_connector(preconfigured_device('LSL'),
                                   ConnectionMethod.LSL, {})
        self.assertTrue(isinstance(connector, LslConnector))
