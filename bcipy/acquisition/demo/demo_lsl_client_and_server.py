"""Sample script to demonstrate usage of LSL client and server."""


def main():
    # pylint: disable=too-many-locals
    """Creates a sample lsl client that reads data from a sample TCP server
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

    from bcipy.acquisition.datastream.generator import random_data_generator
    from bcipy.acquisition.protocols import registry
    from bcipy.acquisition.client import DataAcquisitionClient
    from bcipy.acquisition.datastream.lsl_server import LslDataServer
    from bcipy.acquisition.datastream.tcp_server import await_start

    host = '127.0.0.1'
    port = 9000

    channel_count = 16
    sample_rate = 256
    channels = ['ch{}'.format(c + 1) for c in range(channel_count)]
    # The Protocol is for mocking data.
    server = LslDataServer(
        params={
            'name': 'LSL',
            'channels': channels,
            'hz': sample_rate
        },
        generator=random_data_generator(channel_count=channel_count))
    await_start(server)

    # Device is for reading data.
    # pylint: disable=invalid-name
    Device = registry.find_device('LSL')
    device = Device(connection_params={'host': host, 'port': port})
    raw_data_name = 'demo_raw_data.csv'
    client = DataAcquisitionClient(device=device, raw_data_file_name=raw_data_name)

    try:
        client.start_acquisition()

        print("\nCollecting data for 10s... (Interrupt [Ctl-C] to stop)\n")

        while True:
            time.sleep(10)
            client.stop_acquisition()
            client.cleanup()
            server.stop()
            print(f"The collected data has been written to {raw_data_name}")
            break

    except KeyboardInterrupt:
        print("Keyboard Interrupt; stopping.")
        client.stop_acquisition()
        client.cleanup()
        server.stop()
        print(f"The collected data has been written to {raw_data_name}")


if __name__ == '__main__':
    main()
