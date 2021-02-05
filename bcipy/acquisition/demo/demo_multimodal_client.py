"""Sample script to demonstrate usage of the MultiModalAcquisitionClient with
two continuous data streams with different sample rates."""

from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.protocols.lsl.lsl_connector import LslConnector
from bcipy.acquisition.multi_modal_client import MultiModalDataAcquisitionClient
from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.datastream.tcp_server import await_start
from bcipy.helpers.system_utils import log_to_stdout
from bcipy.acquisition.devices import preconfigured_device
from bcipy.acquisition.demo.demo_eye_tracker_server import (eye_tracker_device,
                                                            eye_tracker_server)
import time
import sys


def main(servers: bool = False, debug: bool = False):
    """Creates a sample multimodal client that reads data from a sample EEG 
    data server and a sample eye tracking data server."""

    if debug:
        log_to_stdout()

    eeg_device = preconfigured_device('DSI-24')
    eeg_connector = LslConnector(connection_params={}, device_spec=eeg_device)
    eye_tracker_connector = LslConnector(connection_params={},
                                         device_spec=eye_tracker_device(),
                                         include_marker_streams=False)

    eye_tracker_data_name = 'demo_eye_tracking_raw_data.csv'
    eeg_data_name = 'demo_eeg_raw_data.csv'

    client = MultiModalDataAcquisitionClient(
        connector=eeg_connector,
        raw_data_file_name=eeg_data_name,
        secondary_connector=eye_tracker_connector,
        secondary_raw_data_file_name=eye_tracker_data_name)

    data_msg = "\nThe collected data has been written to " + \
                f"{eeg_data_name} and {eye_tracker_data_name}"

    if servers:
        eeg_server = LslDataServer(device_spec=eeg_device)
        eye_data_server = eye_tracker_server()
        await_start(eeg_server)
        await_start(eye_data_server)

    try:
        seconds = 3

        print("Waiting for server")
        client.start_acquisition()

        print(
            f"\nCollecting data for {seconds}s... (Interrupt [Ctl-C] to stop)\n"
        )

        time.sleep(1)
        client.marker_writer.push_marker('calibration_trigger')
        time.sleep(seconds - 1)
        # Set logging to debug to determine if offset is reasonable. Expected value would be
        # the time for the secondary client to startup plus 1 second of sleep.
        print(f"Offset: {client.offset}.")
        client.stop_acquisition()
        client.cleanup()
        if servers:
            eeg_server.stop()
            eye_data_server.stop()
        print(data_msg)

    except KeyboardInterrupt:
        print("Keyboard Interrupt; stopping.")
        client.stop_acquisition()
        client.cleanup()
        if servers:
            eeg_server.stop()
            eye_data_server.stop()
        print(data_msg)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--servers', action='store_true')
    parser.add_argument('--debug', action='store_true')
 
    args = parser.parse_args()
    main(args.servers, args.debug)
