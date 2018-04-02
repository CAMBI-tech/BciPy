from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
from datastream import generator, server
from protocols.dsi.dsi_device import DsiDevice
from protocols.dsi.dsi_protocol import DsiProtocol

HOST = '127.0.0.1'
PORT = 9999
connection_params = {'host': HOST, 'port': PORT}


def make_server():
    protocol = DsiProtocol()
    channel_count = len(protocol.channels)
    return server.DataServer(protocol=protocol,
                             generator=generator.random_data,
                             gen_params={'channel_count': channel_count},
                             host=HOST, port=PORT)


def test_device():
    s = make_server()
    s.start()

    _acquisition_init()
    _connect()
    _read_data()

    s.stop()


def test_channels():
    """An exception should be thrown if parameters do not match data read
    from the device."""

    s = make_server()
    s.start()

    device = DsiDevice(connection_params=connection_params,
                       channels=['ch1', 'ch2'])
    assert len(device.channels) == 2
    device.connect()

    with pytest.raises(Exception):
        device.acquisition_init()

    s.stop()


def test_frequency():
    """An exception should be thrown if parameters do not match data read
    from the device."""
    s = make_server()
    s.start()

    device = DsiDevice(connection_params=connection_params, fs=100)
    assert device.fs == 100
    device.connect()

    with pytest.raises(Exception):
        device.acquisition_init()

    s.stop()


def _acquisition_init():
    """Channel and sample rate properties should be updated by reading
    initialization data from the server."""

    device = DsiDevice(connection_params=connection_params, channels=[])
    assert device.fs == DsiDevice.default_fs
    assert len(device.channels) == 0

    device.connect()
    device.acquisition_init()

    assert device.fs == DsiDevice.default_fs
    assert len(device.channels) == len(DsiDevice.default_channels)


def _connect():
    """Should require a connect call before initialization."""

    device = DsiDevice(connection_params=connection_params)

    # Payload size that exceeds sensor_data points should throw an error
    with pytest.raises(AssertionError):
        device.acquisition_init()


def _read_data():
    """Should produce a valid sensor_data record."""

    device = DsiDevice(connection_params=connection_params)

    device.connect()
    device.acquisition_init()
    data = device.read_data()

    assert len(data) > 0
    assert len(data) == len(device.channels)
    for f in data:
        assert isinstance(f, float)
