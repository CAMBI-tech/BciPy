from __future__ import absolute_import, division, print_function

import time
from random import random as rand

import pytest
from acquisition.util import StoppableThread
from protocols.lsl.lsl_device import LslDevice
from pylsl import StreamInfo, StreamOutlet

CHANNEL_COUNT = 8
CHANNELS = ['C3', 'C4', 'Cz', 'FPz', 'POz', 'CPz', 'O1', 'O2']
HZ = 100

connection_params = {}


class Server(StoppableThread):
    '''docstring for Server'''

    def __init__(self, include_meta=False):
        super(Server, self).__init__()

        info = StreamInfo('TestStream', 'EEG', CHANNEL_COUNT,
                          HZ, 'float32', 'uid12345')

        if include_meta:
            channels = info.desc().append_child('channels')
            for c in CHANNELS:
                channels.append_child('channel') \
                    .append_child_value('label', c) \
                    .append_child_value('unit', 'microvolts') \
                    .append_child_value('type', 'EEG')
        self.outlet = StreamOutlet(info)
    # @override ; context manager

    def __enter__(self):
        self.start()
        return self

    # @override ; context manager
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def run(self):
        while self.running():
            sample = [rand() for k in range(CHANNEL_COUNT)]
            self.outlet.push_sample(sample)
            time.sleep(0.1)


def test_minimal_metadata_invariants():
    """Run tests with a server with minimal metadata, missing channel names."""
    with Server(include_meta=False):
        _test_incorrect_number_of_channels()
        _test_incorrect_frequency()
        _test_frequency_init()
        _test_connect()
        _test_read_data()


def test_full_metadata_invariants():
    """Run tests with a server with full metadata including channel names."""
    with Server(include_meta=True):
        _test_mismatched_channel_names()
        _test_channel_init()
        _test_frequency_init()
        _test_connect()
        _test_read_data()


def _test_incorrect_number_of_channels():
    """A list of channels with len that does not match channel_count should
    raise an exception."""

    device = LslDevice(connection_params=connection_params,
                       channels=['ch1', 'ch2'], fs=HZ)
    assert len(device.channels) == 2
    assert len(device.channels) != CHANNEL_COUNT
    device.connect()

    with pytest.raises(Exception):
        device.acquisition_init()


def _test_incorrect_frequency():
    """Provided fs should match sample rate read from device"""
    device = LslDevice(connection_params=connection_params,
                       channels=CHANNELS, fs=300)
    assert device.fs == 300
    device.connect()

    with pytest.raises(Exception):
        device.acquisition_init()


def _test_frequency_init():
    """fs should be initialized from device metadata if not provided"""

    device = LslDevice(connection_params=connection_params,
                       channels=CHANNELS, fs=None)
    device.connect()
    device.acquisition_init()

    assert device.fs == HZ


def _test_connect():
    """Should require a connect call before initialization."""
    device = LslDevice(connection_params=connection_params,
                       channels=[], fs=None)

    with pytest.raises(Exception):
        device.acquisition_init()


def _test_read_data():
    """Should produce a valid data record."""

    device = LslDevice(connection_params=connection_params,
                       channels=CHANNELS, fs=HZ)

    device.connect()
    device.acquisition_init()
    data = device.read_data()

    assert len(data) > 0
    assert len(data) == len(device.channels)
    for f in data:
        assert isinstance(f, float)


def _test_mismatched_channel_names():
    """Provided channel names should match device information."""

    device = LslDevice(connection_params=connection_params,
                       channels=[str(i) for i in range(CHANNEL_COUNT)],
                       fs=HZ)
    assert len(device.channels) == CHANNEL_COUNT
    device.connect()

    with pytest.raises(Exception):
        device.acquisition_init()


def _test_channel_init():
    """Channels should be initialized from device metadata if not provided."""
    device = LslDevice(connection_params=connection_params,
                       channels=[], fs=HZ)
    assert len(device.channels) == 0
    device.connect()
    device.acquisition_init()
    assert len(device.channels) == CHANNEL_COUNT
    assert device.channels == CHANNELS
