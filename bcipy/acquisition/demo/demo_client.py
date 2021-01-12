"""Sample script to demonstrate usage of the DataAcquisitionClient."""


def main():
    """Creates a sample client that reads data from a TCP server
    (see demo/server.py). Data is written to a rawdata.csv file, as well as a
    buffer.db sqlite3 database. These files are written in whichever directory
    the script was run.

    The client can be stopped with a Keyboard Interrupt (Ctl-C)."""

    import time
    import sys
    from psychopy import clock

    # Allow the script to be run from the bci root, acquisition dir, or
    # demo dir.
    sys.path.append('.')
    sys.path.append('..')
    sys.path.append('../..')

    from bcipy.acquisition.client import DataAcquisitionClient
    from bcipy.acquisition.devices import supported_device
    from bcipy.acquisition.connection_method import ConnectionMethod
    from bcipy.acquisition.protocols.dsi.dsi_connector import DsiConnector

    # Start the server with the command:
    # python bcipy/acquisition/datastream/tcp_server.py --name DSI --port 9000
    device_spec = supported_device('DSI')
    connector = DsiConnector(connection_params={
        'host': '127.0.0.1',
        'port': 9000
    },
                             device_spec=device_spec)

    # Use default processor (FileWriter), buffer, and clock.
    client = DataAcquisitionClient(connector=connector, clock=clock.Clock())

    try:
        client.start_acquisition()
        print("\nCollecting data for 3s... (Interrupt [Ctl-C] to stop)\n")
        while True:
            time.sleep(3)
            print(f"Number of samples: {client.get_data_len()}")
            client.stop_acquisition()
            client.cleanup()
            break
    except IOError as e:
        print(f'{e.strerror}; make sure you started the server.')
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        client.stop_acquisition()
        client.cleanup()


if __name__ == '__main__':
    main()
