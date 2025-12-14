"""Demo for the LslAcquisitionClient"""

import time

from bcipy.acquisition import LslAcquisitionClient


def main():
    """Creates a sample client that reads data from an LSL server. The demo
    client is not configured to save data to disk.

    The client can be stopped with a Keyboard Interrupt (Ctl-C)."""

    # Start the server with the command:
    # python bcipy/acquisition/datastream/lsl_server.py --name 'DSI-24'

    client = LslAcquisitionClient(max_buffer_len=1, save_directory='.')

    try:
        seconds = 3
        client.start_acquisition()

        print(
            f"\nCollecting data for {seconds}s... (Interrupt [Ctl-C] to stop)\n"
        )
        time.sleep(seconds)
        samples = client.get_latest_data()
        print(f"Last sample:\n{samples[-1]}")
        print(f"Data written to: {client.recorder.filename}")
        client.stop_acquisition()

    except IOError as err:
        print(f'{err.strerror}; make sure you started the server.')
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        client.stop_acquisition()
        client.cleanup()


if __name__ == '__main__':
    main()
