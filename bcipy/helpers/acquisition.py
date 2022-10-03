"""Helper functions for working with the acquisition module"""
import logging
import subprocess
import time
from typing import Dict, List

import numpy as np

from bcipy.acquisition.devices import DeviceSpec, preconfigured_device
from bcipy.acquisition import LslAcquisitionClient, await_start, LslDataServer
from bcipy.helpers.save import save_device_spec
from bcipy.config import BCIPY_ROOT, RAW_DATA_FILENAME, DEFAULT_DEVICE_SPEC_FILENAME as spec_name

log = logging.getLogger(__name__)


def init_eeg_acquisition(parameters: dict,
                         save_folder: str,
                         export_spec: bool = False,
                         server: bool = False) -> tuple:
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
        export_spec : bool, optional
            if True, the device spec will be exported to the save folder.
        server : bool, optional
            optionally start a mock data server that streams random data.
    Returns
    -------
        (client, server) tuple
    """
    # devices.load(devices_path) can be used in the future to load devices and match
    device_spec = preconfigured_device(parameters['acq_device'])

    dataserver = False
    if server:
        log.info(
            f"fake data is on. Generating mock device data for {device_spec.name}"
        )
        dataserver = LslDataServer(device_spec=device_spec)
        await_start(dataserver)
    else:
        log.info(f"fake data is off. Connecting to {device_spec.name}...")

    client = init_lsl_client(parameters, device_spec, save_folder)
    client.start_acquisition()

    if parameters['acq_show_viewer']:
        viewer_screen = 1 if int(parameters['stim_screen']) == 0 else 0
        start_viewer(display_screen=viewer_screen,
                     parameter_location=parameters['parameter_location'])

    if export_spec:
        save_device_spec(device_spec, save_folder, spec_name)

    return (client, dataserver)


def init_lsl_client(parameters: dict, device_spec: DeviceSpec,
                    save_folder: str):
    """Initialize a client that acquires data from LabStreamingLayer."""

    data_buffer_seconds = round(max_inquiry_duration(parameters))

    return LslAcquisitionClient(max_buffer_len=data_buffer_seconds,
                                device_spec=device_spec,
                                save_directory=save_folder,
                                raw_data_file_name=f'{RAW_DATA_FILENAME}.csv')


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
    stim_count = parameters['stim_length']
    stim_duration = parameters['time_flash']
    interval_duration = parameters['task_buffer_length']
    jitter = parameters['stim_jitter']

    return prestimulus_duration + jitter + target_duration + fixation_duration + (
        stim_count * stim_duration) + interval_duration


def analysis_channels(channels: List[str], device_name: str) -> list:
    """Analysis Channels.

    Defines the channels within a device that should be used for analysis.

    Parameters
    ----------
    - channels(list(str)): list of channel names from the raw_data
    (excluding the timestamp)
    - device_name(str): daq_type from the raw_data file.

    Returns
    --------
    A binary list indicating which channels should be used for analysis.
    If i'th element is 0, i'th channel in filtered_eeg is removed.
    """
    device = preconfigured_device(device_name)
    relevant_channels = device.analysis_channels
    if not relevant_channels:
        raise Exception("Analysis channels for the given device not found: "
                        f"{device_name}.")
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
