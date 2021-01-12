"""Sample script to demonstrate usage of LSL client and server."""


def main(debug: bool = False):
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
    from bcipy.acquisition.devices import DeviceSpec
    from bcipy.acquisition.protocols.lsl.lsl_connector import LslConnector
    from bcipy.acquisition.connection_method import ConnectionMethod
    from bcipy.acquisition.client import DataAcquisitionClient
    from bcipy.acquisition.datastream.lsl_server import LslDataServer
    from bcipy.acquisition.datastream.tcp_server import await_start

    from bcipy.helpers.system_utils import log_to_stdout

    if debug:
        log_to_stdout()

    # Generic LSL device with 16 channels.
    device_spec = DeviceSpec(name="LSL_demo",
                             channels=[
                                 "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6",
                                 "Ch7", "Ch8", "Ch9", "Ch10", "Ch11", "Ch12",
                                 "Ch13", "Ch14", "Ch15", "Ch16"
                             ],
                             sample_rate=256.0)
    server = LslDataServer(device_spec=device_spec)
    await_start(server)

    # Device is for reading data.
    device = LslConnector(connection_params={}, device_spec=device_spec)

    raw_data_name = 'demo_raw_data.csv'
    client = DataAcquisitionClient(connector=device,
                                   raw_data_file_name=raw_data_name)

    try:
        client.start_acquisition()

        print("\nCollecting data for 3s... (Interrupt [Ctl-C] to stop)\n")

        while True:
            time.sleep(1)
            client.marker_writer.push_marker('calibration_trigger')
            time.sleep(2)
            print(
                f"Offset: {client.offset}; since calibration trigger was pushed after 1 second of sleep this value should be close to 1."
            )
            client.stop_acquisition()
            client.cleanup()
            server.stop()
            print(f"\nThe collected data has been written to {raw_data_name}")
            break

    except KeyboardInterrupt:
        print("Keyboard Interrupt; stopping.")
        client.stop_acquisition()
        client.cleanup()
        server.stop()
        print(f"\nThe collected data has been written to {raw_data_name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args.debug)
