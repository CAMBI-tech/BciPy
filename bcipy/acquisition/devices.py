"""Functionality for loading and querying configuration for supported hardware devices."""

import json
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

from bcipy.config import (DEFAULT_ENCODING, DEVICE_SPEC_PATH,
                          SESSION_LOG_FILENAME)

IRREGULAR_RATE: int = 0
DEFAULT_CONFIG = DEVICE_SPEC_PATH
_SUPPORTED_DEVICES: Dict[str, 'DeviceSpec'] = {}
# see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html
SUPPORTED_DATA_TYPES = [
    'float32', 'double64', 'string', 'int32', 'int16', 'int8'
]
DEFAULT_DEVICE_TYPE = 'EEG'
DEFAULT_STATIC_OFFSET = 0.1

logger = logging.getLogger(SESSION_LOG_FILENAME)


class ChannelSpec(NamedTuple):
    """Represents metadata about a channel.

    Attributes:
        name (str): Label in the LSL metadata.
        label (str): Label used within BciPy (raw_data, etc.).
        type (Optional[str]): Type of the channel (e.g., 'EEG', 'Pupil'). Defaults to None.
        units (Optional[str]): Units of measurement for the channel data (e.g., 'microvolts').
                               Defaults to None.
    """
    name: str
    label: str
    type: Optional[str] = None
    units: Optional[str] = None

    def __repr__(self) -> str:
        """Returns a string representation of the ChannelSpec object."""
        fields = ['name', 'label', 'type', 'units']
        items = [(field, self.__getattribute__(field)) for field in fields]
        props = [f"{key}='{val}'" for key, val in items if val]
        return f"ChannelSpec({ ', '.join(props) })"


def channel_spec(channel: Union[str, Dict[str, Any], ChannelSpec]) -> ChannelSpec:
    """Creates a ChannelSpec from the given channel information.

    Args:
        channel (Union[str, Dict[str, Any], ChannelSpec]): Acquisition channel
            information, specified as either just the label (str), a dictionary
            with channel properties, or an existing ChannelSpec object.

    Returns:
        ChannelSpec: A `ChannelSpec` object.

    Raises:
        Exception: If an unexpected channel type is provided.
    """
    if isinstance(channel, str):
        return ChannelSpec(name=channel, label=channel)
    if isinstance(channel, ChannelSpec):
        return channel
    if isinstance(channel, dict):
        return ChannelSpec(**channel)
    raise Exception("Unexpected channel type")


class DeviceStatus(Enum):
    """Represents the recording status of a device during acquisition."""
    ACTIVE = auto()
    PASSIVE = auto()

    def __str__(self) -> str:
        """Returns the lowercase string representation of the DeviceStatus enum member."""
        return self.name.lower()

    @classmethod
    def from_str(cls, name: str) -> 'DeviceStatus':
        """Returns the `DeviceStatus` enum member associated with the given string
        representation.

        Args:
            name (str): The string representation of the device status (e.g., 'active', 'passive').

        Returns:
            DeviceStatus: The corresponding `DeviceStatus` enum member.
        """
        return cls[name.upper()]


class DeviceSpec:
    """Specification for a hardware device used in data acquisition.

    Args:
        name (str): Device short name (e.g., 'DSI-24').
        channels (Union[List[str], List[ChannelSpec], List[dict]]): A list of
            data collection channels. Devices must have at least one channel.
            Channels may be provided as a list of names (str), a list of
            `ChannelSpec` objects, or a list of dictionaries representing channel properties.
        sample_rate (int): Sample frequency in Hz.
        content_type (str, optional): Type of device (e.g., 'EEG', 'MoCap', 'Gaze',
                                      'Audio', 'Markers'). Defaults to `DEFAULT_DEVICE_TYPE` ('EEG').
                                      See https://github.com/sccn/xdf/wiki/Meta-Data.
        description (Optional[str], optional): Device description (e.g.,
                                               'Wearable Sensing DSI-24 dry electrode EEG headset').
                                               Defaults to `name` if not provided.
        excluded_from_analysis (Optional[List[str]], optional): A list of channel labels
                                                                 to exclude from analysis.
                                                                 Defaults to an empty list.
        data_type (str, optional): Data format of a channel. All channels must have the same type.
                                   Defaults to 'float32'. See https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html.
        status (DeviceStatus, optional): Recording status of the device.
                                         Defaults to `DeviceStatus.ACTIVE`.
        static_offset (float, optional): Specifies the static trigger offset (in seconds)
                                         used to align triggers properly with EEG data from LSL.
                                         Defaults to `DEFAULT_STATIC_OFFSET` (0.1).
    """

    def __init__(self,
                 name: str,
                 channels: Union[List[str], List[ChannelSpec], List[dict]],
                 sample_rate: int,
                 content_type: str = DEFAULT_DEVICE_TYPE,
                 description: Optional[str] = None,
                 excluded_from_analysis: Optional[List[str]] = None,
                 data_type: str = 'float32',
                 status: DeviceStatus = DeviceStatus.ACTIVE,
                 static_offset: float = DEFAULT_STATIC_OFFSET):

        assert sample_rate >= 0, "Sample rate can't be negative."
        assert data_type in SUPPORTED_DATA_TYPES

        self.name: str = name
        self.channel_specs: List[ChannelSpec] = [
            channel_spec(ch) for ch in channels]
        self.sample_rate: int = int(sample_rate)
        self.content_type: str = content_type
        self.description: str = description or name
        self.data_type: str = data_type
        self.excluded_from_analysis: List[str] = excluded_from_analysis or []
        self._validate_excluded_channels()
        self.status: DeviceStatus = status
        self.static_offset: float = static_offset

    @property
    def channel_count(self) -> int:
        """Returns the number of channels for the device."""
        return len(self.channel_specs)

    @property
    def channels(self) -> List[str]:
        """Returns a list of channel labels, which may be customized for BciPy."""
        return [ch.label for ch in self.channel_specs]

    @property
    def channel_names(self) -> List[str]:
        """Returns a list of channel names as reported by the device."""
        return [ch.name for ch in self.channel_specs]

    @property
    def analysis_channels(self) -> List[str]:
        """Returns a list of channels used for analysis by the signal module.

        Returns:
            List[str]: A list of channel labels to be used for analysis, excluding
                       any channels specified in `excluded_from_analysis`.
        """

        return list(
            filter(lambda channel: channel not in self.excluded_from_analysis,
                   self.channels))

    @property
    def is_active(self) -> bool:
        """Checks if the device is currently active (recording status set to `DeviceStatus.ACTIVE`).

        Returns:
            bool: True if the device is active, False otherwise.
        """
        return self.status == DeviceStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """Converts the `DeviceSpec` object to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the device's properties.
        """
        return {
            'name': self.name,
            'content_type': self.content_type,
            'channels': [ch._asdict() for ch in self.channel_specs],
            'sample_rate': self.sample_rate,
            'description': self.description,
            'excluded_from_analysis': self.excluded_from_analysis,
            'status': str(self.status),
            'static_offset': self.static_offset
        }

    def __str__(self) -> str:
        """Returns a custom string representation of the `DeviceSpec` object."""
        names = [
            'name', 'content_type', 'channels', 'sample_rate', 'description'
        ]

        def quoted_value(name: str) -> Union[str, Any]:
            value = self.__getattribute__(name)
            return f"'{value}'" if isinstance(value, str) else value

        props = [f"{name}={quoted_value(name)}" for name in names]
        return f"DeviceSpec({ ', '.join(props) })"

    def _validate_excluded_channels(self) -> None:
        """Warns if any excluded channels are not found in the device's channel list.

        This method logs a warning for each channel in `excluded_from_analysis`
        that does not exist in `self.channels`.
        """
        for channel in self.excluded_from_analysis:
            if channel not in self.channels:
                logger.warning(
                    f"Excluded channel {channel} not found in spec for {self.name}"
                )


def make_device_spec(config: Dict[str, Any]) -> DeviceSpec:
    """Constructs a `DeviceSpec` object from a dictionary configuration.

    Args:
        config (Dict[str, Any]): A dictionary containing device configuration parameters.

    Returns:
        DeviceSpec: A `DeviceSpec` object initialized with the provided configuration.

    Raises:
        KeyError: If any required fields are missing in the `config` dictionary.
    """
    default_status = str(DeviceStatus.ACTIVE)
    return DeviceSpec(name=config['name'],
                      content_type=config['content_type'],
                      channels=config['channels'],
                      sample_rate=config['sample_rate'],
                      description=config['description'],
                      excluded_from_analysis=config.get(
                          'excluded_from_analysis', []),
                      status=DeviceStatus.from_str(
                          config.get('status', default_status)),
                      static_offset=config.get('static_offset', DEFAULT_STATIC_OFFSET))


def load(config_path: Path = Path(DEFAULT_CONFIG), replace: bool = False) -> Dict[str, 'DeviceSpec']:
    """Loads the list of supported hardware devices for data acquisition from a
    configuration file.

    Args:
        config_path (Path, optional): Path to the devices JSON file.
                                      Defaults to `Path(DEFAULT_CONFIG)`.
        replace (bool, optional): If True, existing devices are replaced; if False,
                                  values will be overwritten or appended. Defaults to False.

    Returns:
        Dict[str, DeviceSpec]: A dictionary of loaded `DeviceSpec` objects, keyed by device name.
    """
    global _SUPPORTED_DEVICES

    if config_path.is_file():
        with open(config_path, 'r', encoding=DEFAULT_ENCODING) as json_file:
            specs = [make_device_spec(entry) for entry in json.load(json_file)]
            if specs and replace:
                _SUPPORTED_DEVICES.clear()
            for spec in specs:
                _SUPPORTED_DEVICES[spec.name] = spec
    return _SUPPORTED_DEVICES


def preconfigured_devices() -> Dict[str, 'DeviceSpec']:
    """Returns the preconfigured devices, keyed by name.

    If no devices have yet been configured, it loads and returns the `DEFAULT_CONFIG`.

    Returns:
        Dict[str, DeviceSpec]: A dictionary of preconfigured `DeviceSpec` objects.
    """
    global _SUPPORTED_DEVICES
    if not _SUPPORTED_DEVICES:
        load()
    return _SUPPORTED_DEVICES


def preconfigured_device(name: str, strict: bool = True) -> DeviceSpec:
    """Retrieves the `DeviceSpec` with the given name.

    Args:
        name (str): The name of the device to retrieve.
        strict (bool, optional): If True, raises an exception if the device is not found.
                                 Defaults to True.

    Returns:
        DeviceSpec: The `DeviceSpec` object for the specified device.

    Raises:
        ValueError: If `strict` is True and the device is not found.
    """
    device = preconfigured_devices().get(name, None)
    if strict and not device:
        current = ', '.join(
            [f"' {key}'" for key, _ in preconfigured_devices().items()])
        msg = (
            f"Device not found: {name}.\n\n"
            f"The current list of devices includes the following: {current}.\n"
            "You may register new devices using the device module `register` function or in bulk"
            " using `load`.")
        logger.error(msg)
        raise ValueError(msg)
    return device  # type: ignore


def with_content_type(content_type: str) -> List[DeviceSpec]:
    """Retrieves a list of `DeviceSpec` objects with the given content type.

    Args:
        content_type (str): The content type to filter devices by.

    Returns:
        List[DeviceSpec]: A list of `DeviceSpec` objects matching the specified content type.
    """
    return [
        spec for spec in preconfigured_devices().values()
        if spec.content_type == content_type
    ]


def register(device_spec: DeviceSpec) -> None:
    """Registers the given `DeviceSpec`.

    Adds the provided `DeviceSpec` to the collection of preconfigured devices.

    Args:
        device_spec (DeviceSpec): The `DeviceSpec` object to register.
    """
    config = preconfigured_devices()
    config[device_spec.name] = device_spec
