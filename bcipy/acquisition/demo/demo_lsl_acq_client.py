"""Demo for the LslAcquisitionClient"""

import time
from pathlib import Path

from bcipy.acquisition.protocols.lsl.lsl_client import LslAcquisitionClient
from bcipy.helpers.raw_data import load


def main():
    """Creates a sample client that reads data from an LSL server. The demo
    client is not configured to save data to disk.

    The client can be stopped with a Keyboard Interrupt (Ctl-C)."""

    # Start the server with the command:
    # python bcipy/acquisition/datastream/lsl_server.py --name LSL

    client = LslAcquisitionClient(max_buflen=1, save_directory='.')

    try:
        seconds = 3
        client.start_acquisition()


        print(
            f"\nCollecting data for {seconds}s... (Interrupt [Ctl-C] to stop)\n"
        )
        time.sleep(seconds)
        samples = client.get_latest_data()
        print(f"Number of samples: {len(samples)}")
        print(f"Last sample:\n{samples[-1]}")

        filename = Path(client.recorder.filename)

        client.stop_acquisition()

        print(f"First sample time: {client.first_sample_time}")
        print(f"Recorder first sample time: {client.recorder.first_sample_time}")

        data = load(Path(filename))
        frame = data.dataframe
        raw_data_first_sample_stamp = frame.iloc[0]['lsl_timestamp']
        offset = client.first_sample_time - raw_data_first_sample_stamp
        print(f"\nOffset from client start to recording start: {offset:.8f}")

    except IOError as err:
        print(f'{err.strerror}; make sure you started the server.')
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        client.stop_acquisition()
        client.cleanup()


if __name__ == '__main__':
    main()
