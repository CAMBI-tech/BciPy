"""Helper functions for working with the acquisition module"""
import logging
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from bcipy.acquisition import (ClientManager, LslAcquisitionClient,
                               LslDataServer, await_start,
                               discover_device_spec)
from bcipy.acquisition.devices import (DeviceSpec, preconfigured_device,
                                       with_content_type)
from bcipy.config import BCIPY_ROOT
from bcipy.config import DEFAULT_DEVICE_SPEC_FILENAME as spec_name
from bcipy.config import RAW_DATA_FILENAME
from bcipy.helpers.save import save_device_specs

log = logging.getLogger(__name__)


def init_eeg_acquisition(
        parameters: dict,
        save_folder: str,
        server: bool = False) -> Tuple[ClientManager, List[LslDataServer]]:
    """Initialize EEG Acquisition.

    Initializes a client that connects with the EEG data source and begins
    data collection.

    Parameters
    ----------
        parameters : dict
            configuration details regarding the device type and other relevant
            connection information.
             {
               "acq_device": str
             }
        save_folder : str
            path to the folder where data should be saved.
        server : bool, optional
            optionally start a mock data server that streams random data.
            For discovered devices (ex. acquisition_mode: 'EEG'), the first
            matching preconfigured device will be mocked.

    Returns
    -------
        (client_manager, servers) tuple
    """

    servers = []
    manager = ClientManager()

    for stream_type in stream_types(parameters['acq_mode']):
        content_type, device_name = parse_stream_type(stream_type)

        if server:
            server_device_spec = server_spec(content_type, device_name)
            log.info(
                f"Generating mock device data for {server_device_spec.name}")
            dataserver = LslDataServer(server_device_spec)
            servers.append(dataserver)
            # Start the server before init_device so it is discoverable.
            await_start(dataserver)

        device_spec = init_device(content_type, device_name)
        raw_data_name = raw_data_filename(device_spec)

        client = init_lsl_client(parameters, device_spec, save_folder,
                                 raw_data_name)
        manager.add_client(client)

    manager.start_acquisition()

    if parameters['acq_show_viewer']:
        viewer_screen = 1 if int(parameters['stim_screen']) == 0 else 0
        start_viewer(display_screen=viewer_screen,
                     parameter_location=parameters['parameter_location'])

    save_device_specs(manager.device_specs, save_folder, spec_name)

    return (manager, servers)


def raw_data_filename(device_spec: DeviceSpec) -> str:
    """Returns the name of the raw data file for the given device."""
    if device_spec.content_type == 'EEG':
        return f'{RAW_DATA_FILENAME}.csv'

    content_type = '_'.join(device_spec.content_type.split()).lower()
    name = '_'.join(device_spec.name.split()).lower()
    return f"{content_type}_data_{name}.csv"


def init_device(content_type: str,
                device_name: Optional[str] = None) -> DeviceSpec:
    """Initialize a DeviceSpec for the given content type.

    If a device_name is provided, the DeviceSpec will be looked up from the list
    of preconfigured devices, otherwise, BciPy will attempt to discover a device
    with the given type streaming on the network.

    If a discovered device is found in devices.json, the provided configuration
    will override any discovered fields.

    Parameters
    ----------
        content_type - LSL content type (EEG, Gaze, etc).
        device_name - optional; name of the device. If provided, the DeviceSpec
            must be a preconfigured device.
    """
    if device_name:
        return preconfigured_device(device_name, strict=True)
    discovered_spec = discover_device_spec(content_type)
    configured_spec = preconfigured_device(discovered_spec.name, strict=False)
    return configured_spec or discovered_spec


def server_spec(content_type: str,
                device_name: Optional[str] = None) -> LslDataServer:
    """Get the device_spec definition to use for a mock data server. If the
    device_name is provided, the matching preconfigured_device will be used.
    Otherwise, the first matching preconfigured device with the given
    content_type will be used.

    Parameters
    ----------
        content_type - LSL content type (EEG, Gaze, etc).
        device_name - optional; name of the device. If provided, the DeviceSpec
            must be a preconfigured device.
    """
    if device_name:
        return preconfigured_device(device_name, strict=True)
    devices = with_content_type(content_type)
    if not devices:
        raise Exception(
            f"No configured devices have content_type {content_type}.")
    return devices[0]


def parse_stream_type(stream_type: str,
                      delimiter: str = "/") -> Tuple[str, str]:
    """Parses the stream type into a tuple of (content_type, device_name).

    Parameters
    ----------
        stream_type - LSL content type (EEG, Gaze, etc). If you know the name
            of the preconfigured device it can be added 'EEG/DSI-24'

    >>> parse_stream_type('EEG/DSI-24')
    ('EEG', 'DSI-24')
    >>> parse_stream_type('Gaze')
    ('Gaze', None)
    """
    if delimiter in stream_type:
        content_type, device_name = stream_type.split(delimiter)[0:2]
        return (content_type, device_name)
    return (stream_type, None)


def init_lsl_client(parameters: dict,
                    device_spec: DeviceSpec,
                    save_folder: str,
                    raw_data_file_name: str = None):
    """Initialize a client that acquires data from LabStreamingLayer."""

    data_buffer_seconds = round(max_inquiry_duration(parameters))

    return LslAcquisitionClient(max_buffer_len=data_buffer_seconds,
                                device_spec=device_spec,
                                save_directory=save_folder,
                                raw_data_file_name=raw_data_file_name)


def max_inquiry_duration(parameters: dict) -> float:
    """Computes the maximum duration of an inquiry based on the configured
    parameters. Paradigms which don't use all of the settings may have shorter
    durations. This can be used to determine the size of the data buffer.

    Parameters
    ----------
    - parameters : dict
        configuration details regarding the task and other relevant information.
         {
            "time_fixation": float,
            "time_prompt": float,
            "prestim_length": float,
            "stim_length": float,
            "stim_jitter": float,
            "task_buffer_length": float,
            "time_flash": float,

         }

    Returns
    -------
    duration in seconds
    """
    fixation_duration = parameters['time_fixation']
    target_duration = parameters['time_prompt']
    prestimulus_duration = parameters['prestim_length']
    poststim_duration = parameters['prestim_length']
    stim_count = parameters['stim_length']
    stim_duration = parameters['time_flash']
    interval_duration = parameters['task_buffer_length']
    jitter = parameters['stim_jitter']

    return prestimulus_duration + jitter + target_duration + fixation_duration + (
        stim_count * stim_duration) + poststim_duration + interval_duration


def analysis_channels(channels: List[str], device_spec: DeviceSpec) -> list:
    """Analysis Channels.

    Defines the channels within a device that should be used for analysis.

    Parameters
    ----------
    - channels(list(str)): list of channel names from the raw_data
    (excluding the timestamp)
    - device_spec: device from which the data was collected

    Returns
    --------
    A binary list indicating which channels should be used for analysis.
    If i'th element is 0, i'th channel in filtered_eeg is removed.
    """

    relevant_channels = device_spec.analysis_channels
    if not relevant_channels:
        raise Exception("Analysis channels for the given device not found: "
                        f"{device_spec.name}.")
    if channels is None:
        return relevant_channels
    return [int(ch in relevant_channels) for ch in channels]


def analysis_channel_names_by_pos(channels: List[str],
                                  channel_map: List[int]) -> Dict[int, str]:
    """Generate a dict mapping the index of a channel (used in offline
    analysis) to its name.

    Parameters:
    -----------
    - channels : list of channel names
    - channel_map : output of analysis_channels method; binary list
    where each item is either 0 or 1.
    """
    selection = [bool(x) for x in channel_map]
    selected_channels = np.array(channels)[selection]
    return {i: ch for i, ch in enumerate(selected_channels)}


def start_viewer(display_screen: int, parameter_location: str) -> None:
    """Start the data viewer application.

    Parameters
    ----------
    - display_screen : which monitor to use for display; usually 0 or 1.
    - parameter_location : path to parameters.json config file.
    """
    viewer = f'{BCIPY_ROOT}/gui/viewer/data_viewer.py'
    cmd = f'python {viewer} -m {display_screen} -p {parameter_location}'
    subprocess.Popen(cmd, shell=True)

    # hack: wait for window to open, so it doesn't error out when the main
    # window is open fullscreen.
    time.sleep(2)


def stream_types(acq_mode: str, delimiter: str = "+") -> List[str]:
    """Parse the provided acquisition mode into a list of LSL stream types.

    The list of supported/recommended types is available in the XDF wiki:
    https://github.com/sccn/xdf/wiki/Meta-Data

    However, some LSL driver apps deviate from this list so this function does
    not validate that the provided modes are in this list.

    Parameters
    ----------
        acq_mode - delimited list of stream types (ex. 'EEG+Gaze')
        delimiter - optional delimiter; default is '+'
    """
    return list(
        dict.fromkeys([mode.strip() for mode in acq_mode.split(delimiter)]))
