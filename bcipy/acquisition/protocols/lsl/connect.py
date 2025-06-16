"""Utility functions for connecting to an LSL Stream"""
from typing import List, Optional

from pylsl import StreamInfo, resolve_stream

from bcipy.acquisition.devices import DEFAULT_DEVICE_TYPE, DeviceSpec
from bcipy.acquisition.protocols.lsl.lsl_connector import channel_names


def resolve_device_stream(
        device_spec: Optional[DeviceSpec] = None) -> StreamInfo:
    """Resolves and returns the LSL stream for the given device.

    This function searches for an LSL stream based on the `content_type` of the
    provided `DeviceSpec`. If no `DeviceSpec` is provided, it defaults to
    `DEFAULT_DEVICE_TYPE`.

    Args:
        device_spec (Optional[DeviceSpec], optional): The DeviceSpec object
                                                       containing the content type
                                                       of the stream to resolve.
                                                       Defaults to None.

    Returns:
        StreamInfo: The resolved LSL `StreamInfo` object.

    Raises:
        Exception: If an LSL stream with the specified content type is not found.
    """
    content_type = device_spec.content_type if device_spec else DEFAULT_DEVICE_TYPE
    streams = resolve_stream('type', content_type)
    if not streams:
        raise Exception(
            f'LSL Stream not found for content type {content_type}')
    return streams[0]


def device_from_metadata(metadata: StreamInfo) -> DeviceSpec:
    """Creates a `DeviceSpec` object from LSL stream metadata.

    Args:
        metadata (StreamInfo): The LSL `StreamInfo` object containing device metadata.

    Returns:
        DeviceSpec: A `DeviceSpec` object populated with information from the metadata.
    """
    return DeviceSpec(name=metadata.name(),
                      channels=channel_names(metadata),
                      sample_rate=metadata.nominal_srate(),
                      content_type=metadata.type())
