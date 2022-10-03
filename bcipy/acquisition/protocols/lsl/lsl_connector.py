# pylint: disable=fixme
"""Defines the driver for the Device for communicating with
LabStreamingLayer (LSL)."""
import logging
from typing import Dict, List

import pylsl

from bcipy.acquisition.devices import DeviceSpec

log = logging.getLogger(__name__)

LSL_TIMESTAMP = 'LSL_timestamp'
LSL_TIMEOUT_SECONDS = 5.0


class Marker():
    """Data class which wraps a LSL marker; data pulled from a marker stream is
    a tuple where the first item is a list of channels and second item is the
    timestamp. Assumes that marker inlet only has a single channel."""

    def __init__(self, data=(None, None)):
        super(Marker, self).__init__()
        self.channels, self.timestamp = data

    @classmethod
    def empty(cls):
        """Creates an empty Marker."""
        return Marker()

    def __repr__(self):
        return f"<value: {self.trg}, timestamp: {self.timestamp}>"

    @property
    def is_empty(self):
        """Test to see if the current marker is empty."""
        return self.channels is None or self.timestamp is None

    @property
    def trg(self):
        """Get the trigger."""
        # pylint: disable=unsubscriptable-object
        return self.channels[0] if self.channels else None


def inlet_name(inlet) -> str:
    """Returns the name of a pylsl streamInlet."""
    name = '_'.join(inlet.info().name().split())
    return name.replace('-', '_')


def channel_names(stream_info: pylsl.StreamInfo) -> List[str]:
    """Extracts the channel names from the LSL Stream metadata."""
    channels = []
    if stream_info.type() == 'Markers':
        return ['Marker']
    if stream_info.desc().child("channels").empty():
        return channels

    channel = stream_info.desc().child("channels").child("channel")
    for _ in range(stream_info.channel_count()):
        channel_name = channel.child_value("label")
        channels.append(channel_name)
        channel = channel.next_sibling()

    return channels


def check_device(device_spec: DeviceSpec, metadata: pylsl.StreamInfo):
    """Confirm that the properties of the given device_spec match the metadata
    acquired from the device."""
    channels = channel_names(metadata)
    # Confirm that provided channels match metadata, or meta is empty.
    if channels and device_spec.channel_names != channels:
        print(f"device channels: {channels}")
        print(device_spec.channel_names)
        raise Exception("Channels read from the device do not match "
                        "the provided parameters.")
    assert device_spec.channel_count == metadata.channel_count(
    ), "Channel count error"

    if device_spec.sample_rate != metadata.nominal_srate():
        raise Exception("Sample frequency read from device does not match "
                        "the provided parameter")


def rename_items(items: List[str], rules: Dict[str, str]) -> None:
    """Renames items based on the provided rules.
    Parameters
    ----------
        items - list of items ; values will be mutated
        rules - change key -> value
    """
    for key, val in rules.items():
        if key in items:
            items[items.index(key)] = val
