# -*- coding: utf-8 -*-

from __future__ import division

import logging
import acquisition.datastream.generator as generator
import acquisition.protocols.registry as registry
from acquisition.client import Client, _Clock
from acquisition.datastream.server import DataServer
from acquisition.processor import FileWriter


def init_eeg_acquisition(parameters, save_folder,
                         clock=_Clock(), server=False):
    """
    Initializes a client that connects with the EEG data source and begins
    data collection.

    Parameters
    ----------
        parameters : dict
            configuration details regarding the device type and other relevant
            connection information.
             {
               "acq_device": {
                 "value": str
               },
               "acq_host": {
                 "value": str
               },
               "acq_port": {
                 "value": int
               },
               "buffer_name": {
                 "value": str
               },

               "raw_data_name": {
                 "value": str
               }
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
    host = parameters['acq_host']['value']
    port = int(parameters['acq_port']['value'])

    parameters = {
        'buffer_name': save_folder + '/' + parameters['buffer_name']['value'],
        'device': parameters['acq_device']['value'],
        'filename': save_folder + '/' + parameters['raw_data_name']['value'],
        'connection_params': {'host': host,
                              'port': port}}

    # Set configuration parameters (with default values if not provided).
    buffer_name = parameters.get('buffer_name', 'buffer.db')
    connection_params = parameters.get('connection_params', {})
    device_name = parameters.get('device', 'DSI')
    filename = parameters.get('filename', 'rawdata.csv')

    dataserver = False
    if server:
        device_name = 'DSI'
        protocol = registry.default_protocol(device_name)
        dataserver, port = start_socket_server(protocol, host, port)
        connection_params['port'] = port

    Device = registry.find_device(device_name)

    # Start a client. We assume that the channels and fs will be set on the
    # device; add a channel parameter to Device to override!
    client = Client(device=Device(connection_params=connection_params),
                    processor=FileWriter(filename=filename),
                    buffer_name=buffer_name,
                    clock=clock)

    client.start_acquisition()

    # If we're using a server or data generator, there is no reason to
    # calibrate data.
    if server:
        client.is_calibrated = True

    return (client, dataserver)


def start_socket_server(protocol, host, port, retries=2):
    """Starts a DataServer given the provided port and host information. If
    the port is not available, will automatically try a different port up to
    the given number of times. Returns the server along with the port.

    Parameters
    ----------
        protocol : Protocol for how to generate data.
        host : str ; socket host (ex. '127.0.0.1').
        port : int.
        retries : int; number of times to attempt another port if provided
            port is busy.
    Returns
    -------
        (server, port)
    """
    import time

    try:
        dataserver = DataServer(protocol=protocol,
                                generator=generator.random_data,
                                gen_params={'channel_count': len(
                                    protocol.channels)},
                                host=host,
                                port=port)

    except IOError as e:
        if retries > 0:
            # try a different port when 'Address already in use'.
            port = port + 1
            logging.debug("Address in use: trying port {}".format(port))
            return start_socket_server(protocol, host, port, retries - 1)
        else:
            raise e

    dataserver.start()
    # Ensures that server is started before trying to connect to it.
    max_wait = 2  # seconds
    wait = 0
    wait_interval = 0.01
    while not dataserver.started:
        time.sleep(wait_interval)
        wait += wait_interval
        if wait >= max_wait:
            dataserver.stop()
            raise Exception("Server couldn't start up in time.")

    return dataserver, port
