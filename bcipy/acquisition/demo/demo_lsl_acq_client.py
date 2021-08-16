"""Demo for the LslAcquisitionClient"""

import time

from bcipy.acquisition.protocols.lsl.lsl_client import LslAcquisitionClient

def main():
    """Creates a sample client that reads data from an LSL server. The demo
    client is not configured to save data to disk.

    The client can be stopped with a Keyboard Interrupt (Ctl-C)."""

    # Start the server with the command:
    # python bcipy/acquisition/datastream/lsl_server.py --name LSL

    client = LslAcquisitionClient(max_buflen=1, save_directory='.', use_marker_writer=True)

    try:
        seconds = 3
        client.start_acquisition()
        print(
            f"\nCollecting data for {seconds}s... (Interrupt [Ctl-C] to stop)\n"
        )
        client.marker_writer.push_marker('Hello')
        time.sleep(seconds)
        client.marker_writer.push_marker('World')
        samples = client.get_latest_data()
        print(f"Number of samples: {len(samples)}")
        print(f"Last sample:\n{samples[-1]}")

    except IOError as err:
        print(f'{err.strerror}; make sure you started the server.')
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        client.stop_acquisition()
        client.cleanup()


if __name__ == '__main__':
    main()
