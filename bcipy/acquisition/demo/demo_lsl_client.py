"""Sample script to demonstrate usage of the LSL DataAcquisitionClient."""


def main():
    """Creates a sample client that reads data from an LSL server
    (see demo/server.py). Data is written to a rawdata.csv file, as well as a
    buffer.db sqlite3 database. These files are written in whichever directory
    the script was run.

    The client can be stopped with a Keyboard Interrupt (Ctl-C).
    
    Note: This demo assumes that you already have a running lsl_server.
    """

    import sys
    import time

    # Allow the script to be run from the bci root, acquisition dir, or
    # demo dir.
    sys.path.append('.')
    sys.path.append('..')
    sys.path.append('../..')

    from bcipy.acquisition.client import DataAcquisitionClient
    from bcipy.acquisition.protocols import registry
    from bcipy.acquisition.devices import supported_device
    from bcipy.acquisition.connection_method import ConnectionMethod
    from bcipy.acquisition.datastream.lsl_server import MARKER_STREAM_NAME

    # pylint: disable=invalid-name
    device_spec = supported_device('LSL')
    print(f"\nAcquiring data from device {device_spec}")

    Connector = registry.find_device(device_spec, ConnectionMethod.LSL)
    lsl_connector = Connector(connection_params={}, device_spec=device_spec)
    lsl_connector.include_marker_streams = True
    if Connector:
        print(f"Found device connector: {lsl_connector}")

    # Use default processor (FileWriter), buffer, and clock.
    raw_data_name = 'lsl_client_raw_data.csv'
    client = DataAcquisitionClient(device=lsl_connector, raw_data_file_name=raw_data_name)

    try:
        client.start_acquisition()
        print("\nCollecting data for 10s... (Interrupt [Ctl-C] to stop)\n")

        while True:
            time.sleep(10)
            client.stop_acquisition()
            client.cleanup()
            print(f"The collected data has been written to {raw_data_name}")
            break
    except IOError as e:
        print(f'{e.strerror}; make sure you started the server.')
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        print("Number of samples: {0}".format(client.get_data_len()))
        client.stop_acquisition()
        client.cleanup()


if __name__ == '__main__':
    main()
