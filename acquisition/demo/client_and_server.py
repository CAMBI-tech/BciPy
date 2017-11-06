
def main():
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

    import acquisition.datastream.generator as generator
    import acquisition.protocols.registry as registry
    from acquisition.client import Client
    from acquisition.datastream.server import DataServer

    host = '127.0.0.1'
    port = 8844
    # The Protocol is for mocking data.
    protocol = registry.default_protocol('DSI')
    server = DataServer(protocol=protocol,
                        generator=generator.random_data,
                        gen_params={'channel_count': len(protocol.channels)},
                        host=host, port=port)

    # Device is for reading data.
    Device = registry.find_device('DSI')
    dsi_device = Device(connection_params={'host': host, 'port': port})
    client = Client(device=dsi_device)

    try:
        server.start()
        client.start_acquisition()

        print("\nCollecting data... (Interrupt [Ctl-C] to stop)\n")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt; stopping.")
        client.stop_acquisition()
        print("Number of samples: {0}".format(client.get_data_len()))
        server.stop()
        print("The collected data has been written to rawdata.csv")


if __name__ == '__main__':
    main()
