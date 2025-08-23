# pylint: disable=fixme
"""Defines the driver for the Device for communicating with
LabStreamingLayer (LSL)."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import pylsl

from bcipy.acquisition.devices import DeviceSpec
from bcipy.config import SESSION_LOG_FILENAME

log = logging.getLogger(SESSION_LOG_FILENAME)

LSL_TIMESTAMP = 'LSL_timestamp'
LSL_TIMEOUT_SECONDS = 5.0


class Marker:
    """Data class which wraps an LSL marker.

    Data pulled from a marker stream is a tuple where the first item is a list
    of channels (typically one) and the second item is the timestamp.
    Assumes that the marker inlet only has a single channel.

    Args:
        data (Tuple[Optional[List[Any]], Optional[float]], optional): A tuple
            containing the channel data (list) and its timestamp (float).
            Defaults to (None, None).
    """

    def __init__(self, data: Tuple[Optional[List[Any]], Optional[float]] = (None, None)):
        super().__init__()
        self.channels: Optional[List[Any]] = data[0]
        self.timestamp: Optional[float] = data[1]

    @classmethod
    def empty(cls) -> 'Marker':
        """Creates an empty Marker instance.

        Returns:
            Marker: An empty Marker object.
        """
        return Marker()

    def __repr__(self) -> str:
        """Returns a string representation of the Marker object."""
        return f"<value: {self.trg}, timestamp: {self.timestamp}>"

    @property
    def is_empty(self) -> bool:
        """Checks if the current marker is empty.

        Returns:
            bool: True if both channels and timestamp are None, False otherwise.
        """
        return self.channels is None or self.timestamp is None

    @property
    def trg(self) -> Optional[Any]:
        """Gets the trigger value from the marker's channels.

        Assumes the trigger is the first element in the channels list.

        Returns:
            Optional[Any]: The trigger value, or None if channels is empty or None.
        """
        # pylint: disable=unsubscriptable-object
        return self.channels[0] if self.channels else None


def inlet_name(inlet: pylsl.StreamInlet) -> str:
    """Returns a sanitized name of a pylsl `StreamInlet`.

    Converts the stream info name by replacing spaces and hyphens with underscores.

    Args:
        inlet (pylsl.StreamInlet): The LSL StreamInlet object.

    Returns:
        str: The sanitized name of the inlet.
    """
    name = '_'.join(inlet.info().name().split())
    return name.replace('-', '_')


def channel_names(stream_info: pylsl.StreamInfo) -> List[str]:
    """Extracts the channel names from the LSL Stream metadata.

    Args:
        stream_info (pylsl.StreamInfo): The LSL `StreamInfo` object.

    Returns:
        List[str]: A list of channel names. If the stream type is 'Markers',
                   it returns `['Marker']`.
    """
    channels: List[str] = []
    if stream_info.type() == 'Markers':
        return ['Marker']
    if stream_info.desc().child("channels").empty():
        return channels

    channel = stream_info.desc().child("channels").child("channel")
    for _ in range(stream_info.channel_count()):
        channel_name: str = channel.child_value("label")
        channels.append(channel_name)
        channel = channel.next_sibling()

    return channels


def check_device(device_spec: DeviceSpec, metadata: pylsl.StreamInfo) -> None:
    """Confirms that the properties of the given `DeviceSpec` match the metadata
    acquired from the LSL stream.

    Args:
        device_spec (DeviceSpec): The expected `DeviceSpec` for the device.
        metadata (pylsl.StreamInfo): The LSL `StreamInfo` object containing the
                                     actual device metadata.

    Raises:
        Exception: If channel names, channel count, or sample rate do not match
                   between `device_spec` and `metadata`.
    """
    channels = channel_names(metadata)
    # Confirm that provided channels match metadata, or meta is empty.
    if channels and device_spec.channel_names != channels:
        print(f"device channels: {channels}")
        print(device_spec.channel_names)
        raise Exception("Channels read from the device do not match "
                        "the provided parameters.")
    assert device_spec.channel_count == metadata.channel_count(), "Channel count error"

    if device_spec.sample_rate != metadata.nominal_srate():
        raise Exception("Sample frequency read from device does not match "
                        "the provided parameter")


def rename_items(items: List[str], rules: Dict[str, str]) -> None:
    """Renames items in a list based on a provided mapping of rules.

    The list of items is modified in place.

    Args:
        items (List[str]): A list of strings whose values may be mutated.
        rules (Dict[str, str]): A dictionary where keys are original item names
                                and values are their new names.
    """
    for key, val in rules.items():
        if key in items:
            items[items.index(key)] = val
