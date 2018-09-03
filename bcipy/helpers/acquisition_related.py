# -*- coding: utf-8 -*-
from typing import List

import bcipy.acquisition.datastream.generator as generator
import bcipy.acquisition.protocols.registry as registry
from bcipy.acquisition.client import DataAcquisitionClient, _Clock
from bcipy.acquisition.datastream.server import start_socket_server, await_start
from bcipy.acquisition.processor import FileWriter
from bcipy.acquisition.datastream.lsl_server import LslDataServer

# Channels relevant for analysis, for each currently supported device.
#  Note this leaves out triggers or other non-eeg channels. If desired,
#   they should be added to this list.
analysis_channels_by_device = {
    'DSI': ["P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "A1", "Fp1", "Fp2",
            "T3", "T5", "O1", "O2", "F7", "F8", "A2", "T6", "T4"],
    'g.USBamp-2': ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8",
                   "Ch9", "Ch10", "Ch11", "Ch12", "Ch13", "Ch14", "Ch15",
                   "Ch16"],
    'LSL': ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8",
            "ch9", "ch10", "ch11", "ch12", "ch13", "ch14", "ch15", "ch16"]
}


def init_eeg_acquisition(parameters: dict, save_folder: str,
                         clock=_Clock(), server: bool=False):
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

    # Initialize the needed DAQ Parameters
    host = parameters['acq_host']
    port = parameters['acq_port']

    parameters = {
        'buffer_name': save_folder + '/' + parameters['buffer_name'],
        'device': parameters['acq_device'],
        'filename': save_folder + '/' + parameters['raw_data_name'],
        'connection_params': {'host': host,
                              'port': port}}

    # Set configuration parameters (with default values if not provided).
    buffer_name = parameters.get('buffer_name', 'buffer.db')
    connection_params = parameters.get('connection_params', {})
    device_name = parameters.get('device', 'DSI')
    filename = parameters.get('filename', 'rawdata.csv')

    dataserver = False
    if server:
        if device_name == 'DSI':
            protocol = registry.default_protocol(device_name)
            dataserver, port = start_socket_server(protocol, host, port)
            connection_params['port'] = port
        elif device_name == 'LSL':
            channel_count = 16
            sample_rate = 256
            channels = ['ch{}'.format(c + 1) for c in range(channel_count)]
            dataserver = LslDataServer(params={'name': 'LSL',
                                               'channels': channels,
                                               'hz': sample_rate},
                                       generator=generator.random_data(
                                           channel_count=channel_count))
            await_start(dataserver)
        else:
            raise ValueError('Server (fake data mode) for this device type not supported')

    Device = registry.find_device(device_name)

    # Start a client. We assume that the channels and fs will be set on the
    # device; add a channel parameter to Device to override!
    client = DataAcquisitionClient(device=Device(connection_params=connection_params),
                    processor=FileWriter(filename=filename),
                    buffer_name=buffer_name,
                    clock=clock)

    client.start_acquisition()

    # If we're using a server or data generator, there is no reason to
    # calibrate data.
    if server and device_name != 'LSL':
        client.is_calibrated = True

    return (client, dataserver)


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
    relevant_channels = analysis_channels_by_device.get(device_name)
    if not relevant_channels:
        raise Exception("Analysis channels for the given device not found: "
                        f"{device_name}.")
    if channels is None:
        return relevant_channels
    return [int(ch in relevant_channels) for ch in channels]
