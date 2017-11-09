# -*- coding: utf-8 -*-

from __future__ import division

import bci.acquisition.datastream.generator as generator
import bci.acquisition.protocols.registry as registry
from bci.acquisition.buffer import Buffer
from bci.acquisition.client import Client, _Clock
from bci.acquisition.processor import FileWriter
from bci.acquisition.datastream.server import DataServer


def init_eeg_acquisition(parameters, clock=_Clock(), server=True):
    """
    Initializes a client that connects with the EEG data source and begins
    data collection.

    Parameters
    ----------
        parameters : dict
            configuration details regarding the device type and other relevant
            connection information.
                {
                    'buffer_name': str,         # sqlite db path/name
                    'channels': list,           # list of channel names
                    'connection_params': dict,  # device connection params
                    'device': str,              # device name; ex. 'DSI'
                    'filename': str,            # path/name of rawdata file
                    'fs': int                   # sample frequency
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
    default_host = '127.0.0.1'
    default_port = 8844

    # Set configuration parameters (with default values if not provided).
    buffer_name = parameters.get('buffer_name', 'buffer.db')
    channels = parameters.get('channels', [])
    connection_params = parameters.get(
        'connection_params', {'host': default_host, 'port': default_port})
    device_name = parameters.get('device', 'DSI')
    filename = parameters.get('filename', 'rawdata.csv')
    fs = parameters.get('fs', 300)

    dataserver = False
    if server:
        device_name = 'DSI'
        host = connection_params.setdefault('host', default_host)
        port = connection_params.setdefault('port', default_port)
        protocol = registry.default_protocol(device_name)
        fs = protocol.fs
        channels = protocol.channels
        dataserver = DataServer(protocol=protocol,
                                generator=generator.random_data,
                                gen_params={'channel_count': len(channels)},
                                host=host, port=port)
        dataserver.start()

    Device = registry.find_device(device_name)

    client = Client(device=Device(connection_params=connection_params,
                                  fs=fs,
                                  channels=channels),
                    processor=FileWriter.builder(filename),
                    buffer=Buffer.builder(buffer_name),
                    clock=clock)

    return (client, dataserver)
