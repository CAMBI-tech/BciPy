"""Utility functions for connecting to an LSL Stream"""
from typing import Optional

from pylsl import StreamInfo, resolve_stream

from bcipy.acquisition.devices import DEFAULT_DEVICE_TYPE, DeviceSpec
from bcipy.acquisition.protocols.lsl.lsl_connector import channel_names


def resolve_device_stream(
        device_spec: Optional[DeviceSpec] = None) -> StreamInfo:
    """Get the LSL stream for the given device."""
    content_type = device_spec.content_type if device_spec else DEFAULT_DEVICE_TYPE
    streams = resolve_stream('type', content_type)
    if not streams:
        raise Exception(
            f'LSL Stream not found for content type {content_type}')
    return streams[0]


def device_from_metadata(metadata: StreamInfo) -> DeviceSpec:
    """Create a device_spec from the data stream metadata."""
    return DeviceSpec(name=metadata.name(),
                      channels=channel_names(metadata),
                      sample_rate=metadata.nominal_srate(),
                      content_type=metadata.type())
