import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import bcipy.acquisition.protocols.registry as registry
from bcipy.acquisition.client import CountClock, DataAcquisitionClient
from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.datastream.generator import random_data_generator
from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.datastream.tcp_server import TcpDataServer, await_start
from bcipy.acquisition.devices import DeviceSpec, supported_device
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
               "acq_host": str,
               "acq_port": int,
               "buffer_name": str,
               "raw_data_name": str
             }
        clock : Clock, optional
            optional clock used in the client; see client for details.
        server : bool, optional
            optionally start a server that streams random DSI data; defaults
            to true; if this is True, the client will also be a DSI client.
    Returns
    -------
        (client, server) tuple
    """

    # Set configuration parameters with default values if not provided.
    host = parameters['acq_host']
    port = parameters['acq_port']
    buffer_name = Path(save_folder, parameters.get('buffer_name',
                                                   'raw_data.db'))
    raw_data_file = Path(save_folder,
                         parameters.get('raw_data_name', 'raw_data.csv'))

    connection_method = ConnectionMethod.by_name(
        parameters['acq_connection_method'])
    # TODO: parameter for loading devices; path to devices.json?
    # devices.load(devices_path)
    device_spec = supported_device(parameters['acq_device'])

    dataserver = False
    if server:
        dataserver = start_server(connection_method, device_spec, host, port)

    Device = registry.find_device(device_spec, connection_method)
    connection_params = {'host': host, 'port': port}
    device = Device(connection_params=connection_params,
                    device_spec=device_spec)

    client = DataAcquisitionClient(device=device,
                                   buffer_name=buffer_name,
                                   delete_archive=False,
                                   raw_data_file_name=raw_data_file,
                                   clock=clock)

    client.start_acquisition()

    if parameters['acq_show_viewer']:
        viewer_screen = 1 if int(parameters['stim_screen']) == 0 else 0
        start_viewer(display_screen=viewer_screen)

    # If we're using a server or data generator, there is no reason to
    # calibrate data.
    if server and connection_method != ConnectionMethod.LSL:
        client.is_calibrated = True

    return (client, dataserver)


def start_server(connection_method: ConnectionMethod,
                 device_spec: DeviceSpec,
                 host: str = None,
                 port: int = None) -> StoppableThread:
    """Create a server that will generate mock data for the given DeviceSpec
    using the appropriate connection method.
    
    Parameters
    ----------
        connection_method - method used to serve data.
        device_spec - specifies what kind of data to serve (channels, sample_rate, etc).
        host - if using TCP, serves on this host.
        port - if using TCP, serves on this port.
    """
    if connection_method == ConnectionMethod.TCP:
        protocol = registry.find_protocol(device_spec, connection_method)
        dataserver = TcpDataServer(protocol=protocol, host=host, port=port)
    elif connection_method == ConnectionMethod.LSL:
        dataserver = LslDataServer(device_spec=device_spec)
    else:
        raise ValueError(
            f'{connection_method} server (fake data mode) for device type {device_spec.name} not supported'
        )
    await_start(dataserver)
    return dataserver


def analysis_channels(channels: List[str], device_name: str) -> list:
    """Analysis Channels.

    Defines the channels within a device that should be used for analysis.

    Parameters:
    ----------
        channels(list(str)): list of channel names from the raw_data
            (excluding the timestamp)
        device_name(str): daq_type from the raw_data file.
    Returns:
    --------
        A binary list indicating which channels should be used for analysis.
        If i'th element is 0, i'th channel in filtered_eeg is removed.
    """
    device = supported_device(device_name)
    relevant_channels = device.analysis_channels()
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
        channels - list of channel names
        channel_map - output of analysis_channels method; binary list
            where each item is either 0 or 1.
    """
    selection = [bool(x) for x in channel_map]
    selected_channels = np.array(channels)[selection]
    return {i: ch for i, ch in enumerate(selected_channels)}


def start_viewer(display_screen):
    viewer = 'bcipy/gui/viewer/data_viewer.py'
    cmd = f'python {viewer} -m {display_screen}'
    subprocess.Popen(cmd, shell=True)

    # hack: wait for window to open, so it doesn't error out when the main
    # window is open fullscreen.
    time.sleep(2)
