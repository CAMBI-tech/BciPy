# Data Acquisition Module (daq)

The data acquisition module is responsible for interfacing with hardware to obtain device data (EEG, eye tracking, etc). It supports multi-modal recording, real-time querying, and functionality to mock device data for testing.

## Supported Devices

The acquisition module connects with hardware devices using the [Lab Streaming Layer (LSL)](https://labstreaminglayer.readthedocs.io/index.html) library. Each device should provide its own LSL driver / application to use for streaming data. The LSL website maintains a list of [available apps](https://labstreaminglayer.readthedocs.io/info/supported_devices.html). If your device does not have a supported app see [https://labstreaminglayer.readthedocs.io/dev/app_build.html](https://labstreaminglayer.readthedocs.io/dev/app_build.html). The streamer should be started prior to running the `bcipy` application.

Within BciPy users must specify the details of the device they wish to use by providing a `DeviceSpec`. A list of preconfigured devices is defined in `devices.json`. A new device can be added to that file manually, or programmatically registered with the `device_info` module.

    from bcipy.acquisition.devices import DeviceSpec, register
    my_device = DeviceSpec(name="DSI-VR300",
                           channels=["P4", "Fz", "Pz", "F7", "PO8", "PO7", "Oz", "TRG"],
                           sample_rate=300.0,
                           content_type="EEG")
    register(my_device)

## Client

The `lsl_client` module provides the primary interface for interacting with device data and may be used to dynamically query streaming data in real-time. An instance of the `LslClient` class is parameterized with the `DeviceSpec` of interest, as well as the number of seconds of data that should be available for querying. If the `save_directory` and `filename` parameters are provided it also records the data to disk for later offline analysis. If no device is specified the client will attempt to connect to an EEG stream by default. A separate `LslClient` should be constructed for each device of interest.

    from bcipy.acquisition.protocols.lsl.lsl_client import LslAcquisitionClient
    eeg_client = LslAcquisitionClient(max_buflen=1, device_spec=my_device)
    eeg_client.start_acquisition()

### Data queries

Instances of the `'LslClient` can be queried using the `get_data` method. The timestamp parameters passed to this method must be in the same units as the acquisition clock, which uses the pylsl local_clock. If the experiment clock is different that the acquisition clock, the `convert_time` method can be used to get the appropriate value. Only records that are currently buffered are available for return. The buffer size should be set according to the specific needs of an experiment. The entire contents of the buffer can be retrieved using the `get_latest_data` method.

    seconds = 3
    time.sleep(seconds)
    # Get all data currently buffered.
    samples = eeg_client.get_latest_data()
    eeg_client.stop_acquisition()


## Recording Data

The `lsl_recorder` module contains classes for persisting streaming device data to disk. These classes can be used directly if an experiment does not require real-time queries but would still like to record data. The `LslRecorder` listens for all devices streamed by Lab Streaming Layer. To listen to specific devices, an `LslRecordingThread` can be instantiated with a specific device stream name.


## Mock Device Data

The acquisition `datastream` module provides functionality for mocking device data. This is useful for testing and development. The server is initialized with the `DeviceSpec` to mock and an optional generator and writes data to an LSL stream at the sample rate specified in the spec. As with a real device, the `lsl_server` must be started prior to running `bcipy`.

    from bcipy.acquisition.datastream.lsl_server import LslDataServer
    from bcipy.acquisition.devices import preconfigured_device
    server = LslDataServer(device_spec=preconfigured_device('DSI-24'))
    server.start()

If using a preconfigured device the server can be run directly from the command line:

    python bcipy/acquisition/datastream/lsl_server.py --name='DSI-24'

The default settings generate random data, but if a data collection session has been previously recorded the `file_generator` can be used to replay this data. See the `datastream.generator` module for options.

    python bcipy/acquisition/datastream/lsl_server.py --filename='raw_data.csv'

## Alternate DataAcquisitionClient

The `LslClient` is the preferred method for interacting with devices, however, BciPy also provides a more generic client for instances where this may not work, such as connecting to a device through via a TCP interface. This client is more general, but requires more configuration and is not currently supported. Note also that this client was not designed for multi-modal acquisition.

### Example

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

The demo folder contains additional examples of how the various components interact and can be mocked.

### Architecture

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

#### Buffer

The `Buffer` is used by the `DataAcquisitionClient` internally to store data so it can be queried again. The default buffer uses a Sqlite3 database to store data. By default it writes to a file called `buffer.db`, but this can be configured by using the class builder methods. See the client.py main method for an example.

#### Clock

The `Clock` is used to timestamp data as it is received from the device. Data is timestamped immediately before it gets written to the process queue for processing, therefore long-running process steps will not affect subsequent timestamps. A clock must implement the `getTime` method.


## TCP Data Server

A `TcpDataServer` takes a data `generator` and a `Protocol` and streams generated data through a socket connection at the frequency specified in the protocol. Internally a server uses a `Producer` to manage the interval at which data is sent.

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

