"""Example to demonstrate behavior when the StreamInlet max_buflen is exceeded.

Prior to running, start a data streamer, such as the SendData.py example
script included in pylsl.

Example usage:
$ python3 lsl_buffer_test.py --buffer=1 --seconds=2
"""

import time

from pylsl import StreamInlet, resolve_stream


def pull_all_data(inlet: StreamInlet, max_samples: int) -> None:
    """Pull all buffered data in the given stream inlet."""

    print("\nPulling buffered data:")
    count = pull_chunk(inlet, max_samples)
    while count == max_samples:
        count = pull_chunk(inlet, max_samples)


def pull_chunk(inlet: StreamInlet, max_samples: int) -> int:
    """Pull a chunk of samples from a stream inlet.
    Returns the count of samples pulled."""

    _chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=max_samples)
    count = len(timestamps)

    time_range = ""
    if timestamps:
        time_range = f"{round(timestamps[0], 3)} to {round(timestamps[-1], 3)}"

    print(f"-> pulled {count} samples: {time_range};",
          f"sorted: {timestamps == sorted(timestamps)}")

    return count


def main(max_buflen: int, max_samples: int, sleep_seconds: int) -> None:
    """Continuously read from a StreamInlet, pausing for a given time between
    reads and then consuming all data in the stream.
    
    Parameters
    ----------
        max_buflen - StreamInlet max_buflen; buffer size in seconds
        max_samples - maximum samples to pull in a single chunk
        sleep_seconds - seconds to sleep between data pulls
    """

    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    # create a new inlet to read from the stream
    print("\n---")
    print(f"max_buflen seconds: {max_buflen}")
    print(f"max_samples: {max_samples}")
    print(f"sleep_seconds: {sleep_seconds}")
    print("---")
    inlet = StreamInlet(streams[0], max_buflen=max_buflen)

    while True:
        # pull all available samples
        pull_all_data(inlet, max_samples)
        time.sleep(sleep_seconds)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-b',
                        '--buffer',
                        help='StreamInlet max_buflen (seconds)',
                        default=360,
                        type=int)
    parser.add_argument('-c',
                        '--chunk',
                        help='Maximum samples per chunk',
                        default=1024,
                        type=int)
    parser.add_argument('-s',
                        '--seconds',
                        help='Seconds to sleep between data pulls',
                        default=2,
                        type=int)
    args = parser.parse_args()
    main(max_buflen=args.buffer,
         max_samples=args.chunk,
         sleep_seconds=args.seconds)
