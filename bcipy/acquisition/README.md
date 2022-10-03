# Data Acquisition Module (daq)

The data acquisition module is responsible for interfacing with hardware to obtain device data (EEG, eye tracking, etc). It supports multi-modal recording, real-time querying, and functionality to mock device data for testing.

## Supported Devices

The acquisition module connects with hardware devices using the [Lab Streaming Layer (LSL)](https://labstreaminglayer.readthedocs.io/index.html) library. Each device should provide its own LSL driver / application to use for streaming data. The LSL website maintains a list of [available apps](https://labstreaminglayer.readthedocs.io/info/supported_devices.html). If your device does not have a supported app see [https://labstreaminglayer.readthedocs.io/dev/app_build.html](https://labstreaminglayer.readthedocs.io/dev/app_build.html). The streamer should be started prior to running the `bcipy` application.

Within BciPy users must specify the details of the device they wish to use by providing a `DeviceSpec`. A list of preconfigured devices is defined in `devices.json`. A new device can be added to that file manually, or programmatically registered with the `devices` module.

    from bcipy.acquisition.devices import DeviceSpec, register
    my_device = DeviceSpec(name="DSI-VR300",
                           channels=["P4", "Fz", "Pz", "F7", "PO8", "PO7", "Oz", "TRG"],
                           sample_rate=300.0,
                           content_type="EEG")
    register(my_device)

`DeviceSpec` channel data can be provided as a list of channel names or specified as a list of `ChannelSpec` entries. These will be validated against the metadata from the LSL data stream. If provided, `ChannelSpecs` allow users to customize the channel labels and override the values provided by the LSL metadata.

## Client

The `lsl_client` module provides the primary interface for interacting with device data and may be used to dynamically query streaming data in real-time. An instance of the `LslClient` class is parameterized with the `DeviceSpec` of interest, as well as the number of seconds of data that should be available for querying. If the `save_directory` and `filename` parameters are provided it also records the data to disk for later offline analysis. If no device is specified the client will attempt to connect to an EEG stream by default. A separate `LslClient` should be constructed for each device of interest.

    from bcipy.acquisition.protocols.lsl.lsl_client import LslAcquisitionClient
    eeg_client = LslAcquisitionClient(max_buffer_len=1, device_spec=my_device)
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

## Server Details

A data server can be initialized with a `DeviceSpec` and a data `generator`. It is designed to stream generated data at the frequency specified in the spec. Internally a server uses a `Producer` to manage the interval at which data is sent.

#### Generator

Generators are Python functions that yield encoded data. They have a parameter for an
`Encoder`, and may include other parameters. Currently there is a `random_data_generator`, which generates random data, and a `file_data_generator`, which reads through a provided file (ex. a calibration file), and yields one row at a time. The `file_data_generator` is useful for repeatable tests with known data.

#### Producer

A `Producer` is a class internal to a server that manages the generation of data at a specified frequency. It's purpose it to mimic the data rate that would be presented if an actual hardware device was used.
