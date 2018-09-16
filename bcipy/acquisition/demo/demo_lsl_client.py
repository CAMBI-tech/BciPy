
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
    import bcipy.acquisition.protocols.registry as registry

    Device = registry.find_device('LSL')
    dsi_device = Device(connection_params={'host': '127.0.0.1', 'port': 9000})

    # Use default processor (FileWriter), buffer, and clock.
    client = DataAcquisitionClient(device=dsi_device, clock=clock.Clock())

    try:
        client.start_acquisition()
        print("\nCollecting data... (Interrupt [Ctl-C] to stop)\n")
        while True:
            pass
    except IOError as e:
        print(f'{e.strerror}; make sure you started the server.')
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        print("Number of samples: {0}".format(client.get_data_len()))
        client.stop_acquisition()
        client.cleanup()


if __name__ == '__main__':
    main()
