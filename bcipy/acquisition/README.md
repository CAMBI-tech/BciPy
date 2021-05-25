# Data Acquisition Module (daq)

The Data Acquisition module is intended to be used within the Brain Computer Interface (BCI) systems. This module streams data from the EEG hardware, persists it in csv files, and makes it available to other systems.

## Client

The `client` is the primary module for receiving data from the EEG hardware, persisting it to disk, and optionally performing some processing per record.

### Examples

    import time
    from acquisition.client import DataAcquisitionClient
    from bcipy.acquisition.devices import supported_device
    from bcipy.acquisition.protocols.dsi.dsi_connector import DsiConnector

    connector= DsiConnector(connection_params={'host': '127.0.0.1', 'port': 8844},
                            device_spec=supported_device('DSI'))
    client = DataAcquisitionClient(connector=connector)

    try:
        client.start_acquisition()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        print("Number of samples: {0}".format(client.get_data_len()))
        client.stop_acquisition()

The daq/tests/client_test.py file also demonstrates how the various components interact and can be mocked.

### Client Architecture

A `DataAcquisitionClient` is initialized with a `Connector`  and optionally a `Processor`, `Buffer`, and `Clock`. A connector is a driver that knows how to connect to and communicate with specific EEG hardware/software, as well as decode the sensor data. Supported connectors can be queried through the `registry`.

A `DataAcquisitionClient` manages two threads, one for acquisition and one for processing. The acquisition thread continually receives data and writes it to a process queue with an associated timestamp from the `clock`. The process thread watches the queue and sends data to the processor, as well as storing it in the `Buffer` to be queried and archived.

#### Registry

The registry has a list of supported connectors, which can be queried by device and connection method.

    import daq.protocols.registry as registry
    registry.find_connector(device_spec=supported_device('DSI'), 
                            connection_method=ConnectionMethod.TCP)


#### Connector

Connectors are are drivers that knows how to communicate with specific EEG hardware/software. A Connector supports one or more devices over a given `ConnectionMethod`. New connectors can be written by extending the daq.protocols.connector `Connector` class.

A connector can be initialized with a dict of connection params that are relevant to that specific hardware (ex. host and port for socket devices), as well as the specification for the device with which to communicate. If provided, the connector will usually validate the provided parameters against initialization messages received from the device.

BciPy currently has Connectors for reading Wearable Sensing Headsets over TCP (DsiConnector) and for reading EEG devices (including Wearable Sensing and GTec headsets) using LabStreamingLayer (LslConnector). The correct Connector will be automatically chosen based on the Device and ConnectionMethod parameters provided through the GUI.

#### Connection Methods

`ConnectionMethod` is an enumeration of the currently supported connection methods. The currently supported methods are TCP and LabStreamingLayer (LSL).

#### Devices

The `devices` module contains functionality for loading and querying the list of supported hardware devices. A `DeviceSpec` provides the specification for a hardware device used in data acquisition, including the device name, list of channels, sample rate, content_type, and supported connection methods.

#### Processor

Processors extend the daq.processor `Processor` base class. The default processor is the `FileWriter`, which writes each record to a csv file. There is also an `LslProcessor`, which creates an pylsl output stream and writes records to the stream.

#### Buffer

The `Buffer` is used by the `Client` internally to store data so it can be queried again. The default buffer uses a Sqlite3 database to store data. By default it writes to a file called `buffer.db`, but this can be configured by using the class builder methods. See the client.py main method for an example.

#### Clock

The `Clock` is used to timestamp data as it is received from the device. Data is timestamped immediately before it gets written to the process queue for processing, therefore long-running process steps will not affect subsequent timestamps. A clock must implement the `getTime` method.


## Data Server

The `datastream` module is primarily used for testing and development. The main entry point is the `server` module.

### Server Architecture

A Server takes a data `generator` and a `Protocol` and streams generated data  through a socket connection at the frequency specified in the protocol. Internally a server uses a `Producer` to manage the interval at which data is sent.

### Example

    import time

    import acquisition.protocols.registry as registry
    from import acquisition.datastream.generator import random_data_generator, generator_factory
    from acquisition.datastream.tcp_server import TcpDataServer

    protocol = registry.default_protocol('DSI')
    n = len(protocol.channels)

    try:
        server = TcpDataServer(protocol=protocol,
                               generator=generator_factory(random_data_generator, channel_count=n),
                               host='127.0.0.1', port=8844)
        server.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        server.stop()

#### Generator

Generators are Python functions that yield encoded data. They have a parameter for an 
`Encoder`, and may include other parameters. Currently there is a `random_data_generator`, which generates random data, and a `file_data_generator`, which reads through a provided file (ex. a calibration file), and yields one row at a time. The `file_data_generator` is useful for repeatable tests with known data.

#### Protocol

A `Protocol` contains device-specific behavior for how to generate mock data. It contains an `Encoder`, a list of initialization messages, and the sample frequency. An `Encoder` takes an array of floats and generates data in the format appropriate for a given device. For example, the DSI encoder generates binary data.


#### Producer

A `Producer` is a class internal to a server that manages the generation of data at a specified frequency. It's purpose it to mimic the data rate that would be presented if an actual hardware device was used.