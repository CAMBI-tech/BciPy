# pylint: disable=fixme
"""Demo script demonstrating both the acquisition server and client."""


def main(debug: bool = False):
    """Creates a sample client that reads data from a sample TCP server
    (see demo/server.py). Data is written to a rawdata.csv file, as well as a
    buffer.db sqlite3 database. These files are written in whichever directory
    the script was run.

    The client/server can be stopped with a Keyboard Interrupt (Ctl-C)."""

    import time
    import sys

    # Allow the script to be run from the bci root, acquisition dir, or
    # demo dir.
    sys.path.append('.')
    sys.path.append('..')
    sys.path.append('../..')

    from bcipy.acquisition.protocols import registry
    from bcipy.acquisition.client import DataAcquisitionClient
    from bcipy.acquisition.datastream.tcp_server import TcpDataServer
    from bcipy.acquisition.devices import preconfigured_device
    from bcipy.acquisition.connection_method import ConnectionMethod
    from bcipy.helpers.system_utils import log_to_stdout

    if debug:
        log_to_stdout()

    host = '127.0.0.1'
    port = 9000

    device_spec = preconfigured_device('DSI')
    protocol = registry.find_protocol(device_spec, ConnectionMethod.TCP)
    server = TcpDataServer(protocol=protocol, host=host, port=port)

    connection_params = {'host': host, 'port': port}
    connector = registry.make_connector(device_spec, ConnectionMethod.TCP,
                                        connection_params)

    raw_data_name = 'demo_raw_data.csv'
    client = DataAcquisitionClient(connector=connector,
                                   raw_data_file_name=raw_data_name)

    try:
        server.start()
        client.start_acquisition()

        print("\nCollecting data for 3s... (Interrupt [Ctl-C] to stop)\n")

        while True:
            time.sleep(3)
            client.stop_acquisition()
            print(f"\nNumber of samples: {client.get_data_len()}\n")
            client.cleanup()
            server.stop()
            print(
                f"\nThe collected data has been written to {raw_data_name}\n")
            break

    except KeyboardInterrupt:
        print("Keyboard Interrupt; stopping.")
        client.stop_acquisition()
        print(f"\nNumber of samples: {client.get_data_len()}\n")
        client.cleanup()
        server.stop()
        print(f"\nThe collected data has been written to {raw_data_name}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args.debug)
