# Data Acquisition Module (daq)

The Data Acquisition module is intended to be used within the Brain Computer Interface (BCI) systems. This module streams data from the EEG hardware, persists it in csv files, and makes it available to other systems.

## Client

The `client` is the primary module for receiving data from the EEG hardware, persisting it to disk, and optionally performing some processing per record.

### Examples

    import time
    from acquisition.client import DataAcquisitionClient
    import acquisition.protocols.registry as registry

    Device = registry.find_device('DSI')
    dsi_device = Device(connection_params={'host': '127.0.0.1', 'port': 8844})
    # Use default processor (FileWriter), buffer, and clock.
    client = DataAcquisitionClient(device=dsi_device)

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

A `DataAcquisitionClient` is initialized with a `Device`  and optionally a `Processor`, `Buffer`, and `Clock`. A device is a driver that knows how to connect to and communicate with specific EEG hardware/software, as well as decode the sensor data. Supported devices can be queried through the `registry`.

A `DataAcquisitionClient` manages two threads, one for acquisition and one for processing. The acquisition thread continually receives data and writes it to a process queue with an associated timestamp from the `clock`. The process thread watches the queue and sends data to the processor, as well as storing it in the `Buffer` to be queried and archived.

#### Registry

The registry has a list of supported devices. Devices can be queried by name. The device constructor is returned.

    import daq.protocols.registry as registry
    Device = registry.find_device('DSI')


#### Device

Devices are drivers that knows how to connect to and communicate with specific EEG hardware/software. New devices can be written by extending the daq.protocols.device `Device` class.

A device can be initialized with a dict of connection params that are relevant to that specific hardware (ex. host and port for socket devices), as well as the sample frequency and list of channels. If provided the device will usually validate the provided parameters against initialization messages received from the device.

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

    import acquisition.datastream.generator as generator
    import acquisition.protocols.registry as registry
    from acquisition.datastream.server import DataServer

    protocol = registry.default_protocol('DSI')
    n = len(protocol.channels)

    try:
        server = DataServer(protocol=protocol,
                            generator=generator.random_data,
                            gen_params={'channel_count': n},
                            host='127.0.0.1', port=8844)
        server.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        server.stop()

#### Generator

Generators are Python functions that yield encoded data. They have a parameter for an 
`Encoder`, and may include other parameters. Currently there is a `random_generator`, which generates random data, and a `file_generator`, which reads through a provided file (ex. a calibration file), and yields one row at a time. The `file_generator` is useful for repeatable tests with known data.

#### Protocol

A `Protocol` contains device-specific behavior for how to generate mock data. It contains an `Encoder`, a list of initialization messages, and the sample frequency. An `Encoder` takes an array of floats and generates data in the format appropriate for a given device. For example, the DSI encoder generates binary data.


#### Producer

A `Producer` is a class internal to a server that manages the generation of data at a specified frequency. It's purpose it to mimic the data rate that would be presented if an actual hardware device was used.

## sig_pro

sig_pro module defines the sig_pro(input_seq, filt, fs, k) function. In the module is a demo folder which demonstrates the usage of the function. filters.txt is a necessary text file in which filters are stored.

#### Input parameters:

This function processes the raw EEG input through a bandpass filter. Three default filters are hard-coded and can be chosen by specifying the sampling freqeuency of the hardware. Three filters are designed for 256Hz, 300Hz and 1024Hz sampling rates. If another filter is required to be used, it can be passed to the function. Input parameters are:

* ```input_seq```

This parameter is the input multi channel EEG signal. Expected dimensions are Number of Channels x Number of Samples

* ```filt```

Input for using a specific filter. If left empty, according to sampling frequency, a pre-designed filter is going to be used. Filters are pre-designed for fs = 256, 300 or 1024 Hz. For sampling frequencies besides these values, filter needs to be provided to the function.

* ```fs```

Sampling frequency of the hardware in Hz. Default value = 256

* ```k```

Downsampling order. Default value = 2
#### Usage:

Pass an input eeg np.array that is a matrix where every row is a channels data. For example a two channel EEG sequence could be:

```python
input_seq = np.array([[1, 4, ...],
       	               [2, 2, ...]])
```

Specify parameters. If your sampling frequency is different than predefined values, specify the filter. Returned value is another numpy array in the form:

```python
output_seq = np.array([[.3, .4, ...],
       	                [.2, .1, ...]])
```

For other details, refer to demo file or function definition.
