"""Helper functions for working with the acquisition module"""
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

import bcipy.acquisition.protocols.registry as registry
from bcipy.acquisition.client import CountClock, DataAcquisitionClient
from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.datastream.tcp_server import TcpDataServer, await_start
from bcipy.acquisition.devices import DeviceSpec, preconfigured_device
from bcipy.acquisition.protocols.lsl.lsl_client import LslAcquisitionClient
from bcipy.acquisition.util import StoppableThread


def init_eeg_acquisition(parameters: dict,
                         save_folder: str,
                         clock=CountClock(),
                         server: bool = False):
    """Initialize EEG Acquisition.

    Initializes a client that connects with the EEG data source and begins
    data collection.

    Parameters
    ----------
        parameters : dict
            configuration details regarding the device type and other relevant
            connection information.
             {
               "acq_device": str,
               "acq_connection_method": str,
               "acq_host": str,
               "acq_port": int,
               "buffer_name": str,
               "raw_data_name": str
             }
        clock : Clock, optional
            optional clock used in the client; see client for details.
        server : bool, optional
            optionally start a mock data server that streams random data.
    Returns
    -------
        (client, server) tuple
    """

    connection_method = ConnectionMethod.by_name(
        parameters['acq_connection_method'])

    # TODO: parameter for loading devices; path to devices.json?
    # devices.load(devices_path)
    device_spec = preconfigured_device(parameters['acq_device'])

    dataserver = False
    if server:
        dataserver = start_server(connection_method, device_spec,
                                  parameters['acq_host'],
                                  parameters['acq_port'])

    client = init_client(connection_method, parameters, device_spec,
                         save_folder)
    client.start_acquisition()

    if parameters['acq_show_viewer']:
        viewer_screen = 1 if int(parameters['stim_screen']) == 0 else 0
        start_viewer(display_screen=viewer_screen,
                     parameter_location=parameters['parameter_location'])

    # If we're using a server or data generator, there is no reason to
    # calibrate data.
    if server and connection_method != ConnectionMethod.LSL:
        client.is_calibrated = True

    return (client, dataserver)


def init_client(connection_method: ConnectionMethod, parameters: dict,
                device_spec: DeviceSpec, save_folder: str):
    """Initialize the correct acquisition client based on the connection
    method."""
    if connection_method == ConnectionMethod.TCP:
        warnings.warn("TCP connections are no longer supported",
                      DeprecationWarning)
        return init_tcp_client(parameters, device_spec, save_folder)
    # return init_lsl_client(parameters, device_spec, save_folder)
    return init_lsl_client_original(parameters, device_spec, save_folder)


def init_tcp_client(parameters: dict, device_spec: DeviceSpec,
                    save_folder: str) -> DataAcquisitionClient:
    """Initialize a TCP client."""

    connector = registry.make_connector(device_spec, ConnectionMethod.TCP, {
        'host': parameters['acq_host'],
        'port': parameters['acq_port']
    })

    return DataAcquisitionClient(
        connector=connector,
        buffer_name=str(
            Path(save_folder, parameters.get('buffer_name', 'raw_data.db'))),
        delete_archive=False,
        raw_data_file_name=Path(
            save_folder, parameters.get('raw_data_name', 'raw_data.csv')))


def init_lsl_client_original(parameters: dict, device_spec: DeviceSpec,
                             save_folder: str) -> DataAcquisitionClient:
    """Initialize a TCP client."""

    connector = registry.make_connector(device_spec, ConnectionMethod.LSL, {
        'host': parameters['acq_host'],
        'port': parameters['acq_port']
    })

    return DataAcquisitionClient(
        connector=connector,
        buffer_name=str(
            Path(save_folder, parameters.get('buffer_name', 'raw_data.db'))),
        delete_archive=False,
        raw_data_file_name=Path(
            save_folder, parameters.get('raw_data_name', 'raw_data.csv')))


def init_lsl_client(parameters: dict, device_spec: DeviceSpec,
                    save_folder: str):
    """Initialize a client that acquires data from LabStreamingLayer."""

    data_buffer_seconds = round(max_inquiry_duration(parameters))

    return LslAcquisitionClient(max_buflen=data_buffer_seconds,
                                device_spec=device_spec,
                                save_directory=save_folder,
                                raw_data_file_name=parameters.get(
                                    'raw_data_name', None),
                                use_marker_writer=True)


def max_inquiry_duration(parameters: dict) -> float:
    """Computes the maximum duration of an inquiry based on the configured
    parameters. Modes which don't use all of the settings may have shorter
    durations.

    Returns
    -------
    duration in seconds
    """
    fixation_duration = parameters['time_cross']
    target_duration = parameters['time_target']
    stim_count = parameters['stim_length']
    stim_duration = parameters['time_flash']
    interval_duration = parameters['task_buffer_len']

    return target_duration + fixation_duration + (
        stim_count * stim_duration) + interval_duration


def start_server(connection_method: ConnectionMethod,
                 device_spec: DeviceSpec,
                 host: str = None,
                 port: int = None) -> StoppableThread:
    """Create a server that will generate mock data for the given DeviceSpec
    using the appropriate connection method.

    Parameters
    ----------
    - connection_method : method used to serve data.
    - device_spec : specifies what kind of data to serve (channels, sample_rate, etc).
    - host : if using TCP, serves on this host.
    - port : if using TCP, serves on this port.
    """
    if connection_method == ConnectionMethod.TCP:
        protocol = registry.find_protocol(device_spec, connection_method)
        dataserver = TcpDataServer(protocol=protocol, host=host, port=port)
    elif connection_method == ConnectionMethod.LSL:
        dataserver = LslDataServer(device_spec=device_spec)
    else:
        raise ValueError(
            (f'{connection_method} server (fake data mode) for device type '
             f'{device_spec.name} not supported'))
    await_start(dataserver)
    return dataserver


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
    viewer = 'bcipy/gui/viewer/data_viewer.py'
    cmd = f'python {viewer} -m {display_screen} -p {parameter_location}'
    subprocess.Popen(cmd, shell=True)

    # hack: wait for window to open, so it doesn't error out when the main
    # window is open fullscreen.
    time.sleep(2)
