"""Sample script to demonstrate usage of the LSL DataAcquisitionClient."""
import sys
import time

from bcipy.acquisition.client import DataAcquisitionClient
from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.datastream.tcp_server import await_start
from bcipy.acquisition.demo.demo_eye_tracker_server import (eye_tracker_device,
                                                            eye_tracker_server)
from bcipy.acquisition.protocols.lsl.lsl_connector import LslConnector


def main():
    """Creates a sample client that reads data from a demo LSL server
    serving mock eye tracking data.
    """

    device_spec = eye_tracker_device()
    print(f"\nAcquiring data from device {device_spec}")
    connector = LslConnector(connection_params={},
                             device_spec=device_spec,
                             include_marker_streams=False,
                             include_lsl_timestamp=True)

    raw_data_name = 'demo_eye_tracking_raw_data.csv'
    client = DataAcquisitionClient(connector=connector,
                                   raw_data_file_name=raw_data_name)

    server = eye_tracker_server()
    await_start(server)
    try:
        client.start_acquisition()
        print("\nCollecting data for 3s... (Interrupt [Ctl-C] to stop)\n")

        while True:
            time.sleep(3)
            print(f"Number of samples: {client.get_data_len()}")
            client.stop_acquisition()
            client.cleanup()
            server.stop()
            print(f"The collected data has been written to {raw_data_name}")
            break
    except IOError as e:
        print(f'{e.strerror}; make sure you started the server.')
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        client.stop_acquisition()
        client.cleanup()
        server.stop()


if __name__ == '__main__':
    main()
