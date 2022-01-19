"""Sample script to demonstrate usage of the LSL DataAcquisitionClient."""
import time

from bcipy.acquisition.protocols.lsl.lsl_client import LslAcquisitionClient
from bcipy.acquisition.datastream.tcp_server import await_start
from bcipy.acquisition.datastream.mock.eye_tracker_server import (eye_tracker_device,
                                                                  eye_tracker_server)


def main():
    """Creates a sample client that reads data from a demo LSL server
    serving mock eye tracking data.
    """

    device_spec = eye_tracker_device()

    raw_data_name = 'demo_eye_tracking_raw_data.csv'
    client = LslAcquisitionClient(max_buflen=1,
                                  device_spec=device_spec,
                                  save_directory='.',
                                  raw_data_file_name=raw_data_name)

    server = eye_tracker_server()
    await_start(server)
    try:
        client.start_acquisition()
        print("\nCollecting data for 3s... (Interrupt [Ctl-C] to stop)\n")

        while True:
            time.sleep(3)
            print("Stopping acquisition")
            client.stop_acquisition()
            client.cleanup()
            print("Stopping server")
            server.stop()
            print(f"The collected data has been written to {raw_data_name}")
            break
    except IOError as err:
        print(f'{err.strerror}; make sure you started the server.')
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        client.stop_acquisition()
        client.cleanup()
        server.stop()


if __name__ == '__main__':
    main()
