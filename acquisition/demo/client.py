
def main():
    """Creates a sample client that reads data from a TCP server
    (see demo/server.py). Data is written to a rawdata.csv file, as well as a
    buffer.db sqlite3 database. These files are written in whichever directory
    the script was run.

    The client can be stopped with a Keyboard Interrupt (Ctl-C)."""

    import time
    import sys

    # Allow the script to be run from the bci root, acquisition dir, or
    # demo dir.
    sys.path.append('.')
    sys.path.append('..')
    sys.path.append('../..')

    from acquisition.client import Client
    import acquisition.protocols.registry as registry

    Device = registry.find_device('DSI')
    dsi_device = Device(connection_params={'host': '0.0.0.0', 'port': 8844})

    # Use default processor (FileWriter), buffer, and clock.
    client = Client(device=dsi_device)

    try:
        client.start_acquisition()
        print("\nCollecting data... (Interrupt [Ctl-C] to stop)\n")
        while True:
            time.sleep(1)
    except IOError as e:
        print "{0}; make sure you started the server.".format(e.strerror)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        print("Number of samples: {0}".format(client.get_data_len()))
        client.stop_acquisition()


if __name__ == '__main__':
    main()
