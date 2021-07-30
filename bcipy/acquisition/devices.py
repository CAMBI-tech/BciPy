"""Functionality for loading and querying configuration for supported hardware
devices."""
import json
from pathlib import Path
from typing import Dict, List

from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.helpers.system_utils import auto_str

DEFAULT_CONFIG = 'bcipy/acquisition/devices.json'
_SUPPORTED_DEVICES = {}


@auto_str
class DeviceSpec:
    """Specification for a hardware device used in data acquisition.

    Parameters
    ----------
        name - device short name; ex. DSI-24
        channels - list of data collection channels; devices must have at least
            one channel.
        sample_rate - sample frequency in Hz.
        content_type - type of device; likely one of ['EEG', 'MoCap', 'Gaze',
            'Audio', 'Markers']; see https://github.com/sccn/xdf/wiki/Meta-Data.
        connection_methods - list of methods for connecting to the device
        description - device description
            ex. 'Wearable Sensing DSI-24 dry electrode EEG headset'
        excluded_from_analysis - list of channels to exclude from analysis.
    """

    def __init__(self,
                 name: str,
                 channels: List[str],
                 sample_rate: float,
                 content_type: str = 'EEG',
                 connection_methods: List[ConnectionMethod] = None,
                 description: str = None,
                 excluded_from_analysis: List[str] = ['TRG']):
        self.name = name
        self.channels = channels
        self.sample_rate = sample_rate
        self.content_type = content_type
        self.connection_methods = connection_methods or [ConnectionMethod.LSL]
        self.description = description or name
        self.excluded_from_analysis = excluded_from_analysis

    @property
    def channel_count(self) -> int:
        return len(self.channels)

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



def make_device_spec(config: dict) -> DeviceSpec:
    """Constructs a DeviceSpec from a dict. Throws a KeyError if any fields
    are missing."""
    connection_methods = list(
        map(ConnectionMethod.by_name, config.get('connection_methods', [])))
    return DeviceSpec(name=config['name'],
                      content_type=config['content_type'],
                      channels=config['channels'],
                      connection_methods=connection_methods,
                      sample_rate=config['sample_rate'],
                      description=config['description'])


def load(config_path: Path = Path(DEFAULT_CONFIG)) -> Dict[str, DeviceSpec]:
    """Load the list of supported hardware for data acquisition from the given
    configuration file."""
    global _SUPPORTED_DEVICES
    with open(config_path, 'r', encoding='utf-8') as json_file:
        specs = [make_device_spec(entry) for entry in json.load(json_file)]
        _SUPPORTED_DEVICES = {spec.name: spec for spec in specs}


def supported_devices() -> Dict[str, DeviceSpec]:
    """Returns the currently supported devices keyed by name."""
    global _SUPPORTED_DEVICES
    if not _SUPPORTED_DEVICES:
        load()
    return _SUPPORTED_DEVICES


def supported_device(name: str) -> DeviceSpec:
    """Retrieve the DeviceSpec with the given name. An exception is raised
    if the device is not supported."""
    device = supported_devices().get(name, None)
    if not device:
        raise ValueError(f"Device not supported: {name}")
    return device


def register(device_spec: DeviceSpec):
    """Register the given DeviceSpec."""
    config = supported_devices()
    config[device_spec.name] = device_spec
