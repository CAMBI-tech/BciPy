"""Sample script to demonstrate usage of LSL client and server."""
import subprocess
import time

from bcipy.acquisition.client import DataAcquisitionClient
from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.datastream.tcp_server import await_start
from bcipy.acquisition.devices import IRREGULAR_RATE, DeviceSpec
from bcipy.acquisition.protocols.lsl.lsl_connector import LslConnector
from bcipy.helpers.system_utils import log_to_stdout


def start_switch():
    """Start the demo switch"""
    return subprocess.Popen('python bcipy/acquisition/demo/demo_switch.py', shell=True)


def main(debug: bool = False):
    # pylint: disable=too-many-locals
    """Creates a sample lsl client that reads data from a sample TCP server
    (see demo/server.py). Data is written to a rawdata.csv file, as well as a
    buffer.db sqlite3 database. These files are written in whichever directory
    the script was run.

    The client/server can be stopped with a Keyboard Interrupt (Ctl-C)."""

    if debug:
        log_to_stdout()

    # Generic LSL device with 16 channels.
    eeg_device = DeviceSpec(name="LSL_demo",
                            channels=[
                                "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6",
                                "Ch7", "Ch8", "Ch9", "Ch10", "Ch11", "Ch12",
                                "Ch13", "Ch14", "Ch15", "Ch16"
                            ],
                            sample_rate=300.0)
    eeg_server = LslDataServer(device_spec=eeg_device)

    # button_device = DeviceSpec(name="TestButton",
    #                            channels=['Btn1'],
    #                            sample_rate=IRREGULAR_RATE,
    #                            content_type="Button")

    connector = LslConnector(connection_params={},
                             device_spec=eeg_device,
                             include_marker_streams=True)

    raw_data_name = 'multimodal_raw_data.csv'
    client = DataAcquisitionClient(connector=connector,
                                   raw_data_file_name=raw_data_name)

    await_start(eeg_server)

    # Open the Demo Switch GUI.
    pid = start_switch()
    # Wait for switch start
    print("Waiting for Switch")
    time.sleep(0.5)
    try:
        seconds = 5
        client.start_acquisition()
        print(f"\nCollecting data for {seconds}s...",
              "Click in the demo switch GUI to register a switch hit.",
              "Close the GUI when finished.\n")

        time.sleep(1)
        client.marker_writer.push_marker('calibration_trigger')

        time.sleep(seconds - 1)

        # since calibration trigger was pushed after 1 second of sleep
        print(f"Offset: {client.offset}; this value should be close to 1.")
        client.stop_acquisition()
        client.cleanup()
        eeg_server.stop()
        print(f"\nThe collected data has been written to {raw_data_name}")

    except KeyboardInterrupt:
        print("Keyboard Interrupt; stopping.")
        client.stop_acquisition()
        client.cleanup()
        eeg_server.stop()
        print(f"\nThe collected data has been written to {raw_data_name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args.debug)
