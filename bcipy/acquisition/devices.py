"""Functionality for loading and querying configuration for supported hardware
devices."""
import json
import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Union

from bcipy.config import DEFAULT_ENCODING, DEVICE_SPEC_PATH

IRREGULAR_RATE = 0.0
DEFAULT_CONFIG = DEVICE_SPEC_PATH
_SUPPORTED_DEVICES = {}
# see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html
SUPPORTED_DATA_TYPES = [
    'float32', 'double64', 'string', 'int32', 'int16', 'int8'
]
DEFAULT_DEVICE_TYPE = 'EEG'

log = logging.getLogger(__name__)


class ChannelSpec(NamedTuple):
    """Represents metadata about a channel."""
    name: str  # Label in the LSL metadata
    label: str  # Label used within BciPy (raw_data, etc.)
    type: str = None
    units: str = None

    def __repr__(self):
        fields = ['name', 'label', 'type', 'units']
        items = [(field, self.__getattribute__(field)) for field in fields]
        props = [f"{key}='{val}'" for key, val in items if val]
        return f"ChannelSpec({', '.join(props)})"


def channel_spec(channel: Union[str, dict, ChannelSpec]) -> ChannelSpec:
    """Creates a ChannelSpec from the given channel.

    Parameters
    ----------
        channel - acquisition channel information specified as either just the
          label or with additional data represented by a dict or ChannelSpec.
    """
    if isinstance(channel, str):
        return ChannelSpec(name=channel, label=channel)
    if isinstance(channel, ChannelSpec):
        return channel
    if isinstance(channel, dict):
        return ChannelSpec(**channel)
    raise Exception("Unexpected channel type")


class DeviceSpec:
    """Specification for a hardware device used in data acquisition.

    Parameters
    ----------
        name - device short name; ex. DSI-24
        channels - list of data collection channels; devices must have at least
            one channel. Channels may be provided as a list of names or list of
            ChannelSpecs.
        sample_rate - sample frequency in Hz.
        content_type - type of device; likely one of ['EEG', 'MoCap', 'Gaze',
            'Audio', 'Markers']; see https://github.com/sccn/xdf/wiki/Meta-Data.
        description - device description
            ex. 'Wearable Sensing DSI-24 dry electrode EEG headset'
        data_type - data format of a channel; all channels must have the same type;
            see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html
        excluded_from_analysis - list of channels (label) to exclude from analysis.
    """

    def __init__(self,
                 name: str,
                 channels: Union[List[str], List[ChannelSpec], List[dict]],
                 sample_rate: float,
                 content_type: str = DEFAULT_DEVICE_TYPE,
                 description: str = None,
                 excluded_from_analysis: List[str] = None,
                 data_type='float32'):

        assert sample_rate >= 0, "Sample rate can't be negative."
        assert data_type in SUPPORTED_DATA_TYPES

        self.name = name
        self.channel_specs = [channel_spec(ch) for ch in channels]
        self.sample_rate = sample_rate
        self.content_type = content_type
        self.description = description or name
        self.data_type = data_type
        self.excluded_from_analysis = excluded_from_analysis or []
        self._validate_excluded_channels()

    @property
    def channel_count(self) -> int:
        """Number of channels"""
        return len(self.channel_specs)

    @property
    def channels(self) -> List[str]:
        """List of channel labels. These may be customized for BciPy."""
        return [ch.label for ch in self.channel_specs]

    @property
    def channel_names(self) -> List[str]:
        """List of channel names from the device."""
        return [ch.name for ch in self.channel_specs]

    @property
    def analysis_channels(self) -> List[str]:
        """List of channels used for analysis by the signal module.
        Parameters:
        -----------
            exclude_trg - indicates whether or not to exclude a TRG channel if present.
        """

        return list(
            filter(lambda channel: channel not in self.excluded_from_analysis,
                   self.channels))

    def to_dict(self) -> dict:
        """Converts the DeviceSpec to a dict."""
        return {
            'name': self.name,
            'content_type': self.content_type,
            'channels': [ch._asdict() for ch in self.channel_specs],
            'sample_rate': self.sample_rate,
            'description': self.description,
            'excluded_from_analysis': self.excluded_from_analysis
        }

    def __str__(self):
        """Custom str representation."""
        names = [
            'name', 'content_type', 'channels', 'sample_rate', 'description'
        ]

        def quoted_value(name):
            value = self.__getattribute__(name)
            return f"'{value}'" if isinstance(value, str) else value

        props = [f"{name}={quoted_value(name)}" for name in names]
        return f"DeviceSpec({', '.join(props)})"

    def _validate_excluded_channels(self):
        """Warn if excluded channels are not in the list of channels"""
        for channel in self.excluded_from_analysis:
            if channel not in self.channels:
                log.warning(
                    f"Excluded channel {channel} not found in spec for {self.name}"
                )


def make_device_spec(config: dict) -> DeviceSpec:
    """Constructs a DeviceSpec from a dict. Throws a KeyError if any fields
    are missing."""
    return DeviceSpec(name=config['name'],
                      content_type=config['content_type'],
                      channels=config['channels'],
                      sample_rate=config['sample_rate'],
                      description=config['description'],
                      excluded_from_analysis=config.get(
                          'excluded_from_analysis', []))


def load(config_path: Path = Path(DEFAULT_CONFIG)) -> Dict[str, DeviceSpec]:
    """Load the list of supported hardware for data acquisition from the given
    configuration file."""
    global _SUPPORTED_DEVICES
    with open(config_path, 'r', encoding=DEFAULT_ENCODING) as json_file:
        specs = [make_device_spec(entry) for entry in json.load(json_file)]
        _SUPPORTED_DEVICES = {spec.name: spec for spec in specs}
        return _SUPPORTED_DEVICES


def preconfigured_devices() -> Dict[str, DeviceSpec]:
    """Returns the preconfigured devices keyed by name."""
    global _SUPPORTED_DEVICES
    if not _SUPPORTED_DEVICES:
        load()
    return _SUPPORTED_DEVICES


def preconfigured_device(name: str) -> DeviceSpec:
    """Retrieve the DeviceSpec with the given name. An exception is raised
    if the device is not found."""
    device = preconfigured_devices().get(name, None)
    if not device:
        raise ValueError(f"Device not found: {name}")
    return device


def register(device_spec: DeviceSpec):
    """Register the given DeviceSpec."""
    config = preconfigured_devices()
    config[device_spec.name] = device_spec
