"""Records LSL data streams to a data store."""
import logging
import time
from pathlib import Path
from pylsl import StreamInfo, StreamInlet, resolve_streams

from bcipy.acquisition.util import StoppableThread
from bcipy.acquisition.protocols.lsl.lsl_connector import channel_names
from bcipy.helpers.raw_data import RawDataWriter

log = logging.getLogger(__name__)


class LslRecorder:
    """Records LSL data to a datastore. Resolves streams when started.

    Parameters:
    -----------
        path - location to store the recordings

    """

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.streams = None

    def start(self):
        """Start recording all streams currently on the network."""

        if not self.streams:
            log.debug("Recording data")
            # create a thread for each.
            self.streams = [
                LslRecordingThread(stream, self.path)
                for stream in resolve_streams()
            ]

            # Validate that streams have unique names for their type
            stream_names = [stream.filename for stream in self.streams]
            if len(stream_names) != len(set(stream_names)):
                raise Exception("Data stream names are not unique")
            for stream in self.streams:
                stream.start()

    def stop(self, wait: bool = False):
        """Stop recording

        Parameters
        ----------
        - wait : if True waits for all threads to stop before returning.
        """
        for stream in self.streams:
            stream.stop()
            if wait:
                stream.join()
        self.streams = None


class LslRecordingThread(StoppableThread):
    """Records data for the given LabStreamingLayer data stream.

    Parameters:
    ----------
    - stream : information about the stream of interest
    - path : location to store the recording
    """

    def __init__(self, stream_info: StreamInfo, path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_info = stream_info
        self.path = path

        self.sample_count = 0
        self.max_chunk_size = 1024

        # seconds to sleep between data pulls from LSL
        self.sleep_seconds = 0.2
        self.writer = None

    @property
    def filename(self) -> str:
        """Filename, including the path of the raw_data to be written."""
        # TODO: EEG content should take a filename parameter.
        content_type = '_'.join(self.stream_info.type().split()).lower()
        name = '_'.join(self.stream_info.name().split()).lower()

        return str(Path(self.path, f"{content_type}_data_{name}.csv"))

    def init_data_store(self, metadata: StreamInfo):
        """Opens a csv file for the raw_data format and writes the metadata
        headers.

        Parameters:
        ----------
            metadata - full metadata that is the result of a stream_inlet.info()
                call; self.stream_info does not have the channel names
        """
        assert self.writer is None
        filename = self.filename
        log.debug(f"Writing data to {filename}")
        self.writer = RawDataWriter(
            filename,
            daq_type=self.stream_info.name(),
            sample_rate=self.stream_info.nominal_srate(),
            columns=['timestamp'] + channel_names(metadata) +
            ['lsl_timestamp'])
        self.writer.__enter__()

    def cleanup(self):
        """Performs cleanup tasks."""
        assert self.writer, "Writer not initialized"
        self.writer.__exit__()

    def write_chunk(self, data, timestamps):
        """Persists the data."""
        assert self.writer, "Writer not initialized"

        log.debug(f"Writing {len(timestamps)} entries")
        chunk = []
        for i, sample in enumerate(data):
            self.sample_count += 1
            chunk.append([self.sample_count] + sample + [timestamps[i]])
        self.writer.writerows(chunk)

    # @override
    def run(self):
        """Process startup. Connects to the device and start reading data.
        Since this is done in a separate process from the main thread, any
        errors encountered will be written to the msg_queue.
        """
        # Note that self.stream_info does not have the channel names.
        inlet = StreamInlet(self.stream_info)
        full_metadata = inlet.info()

        log.debug("Acquiring data from data stream:")
        log.debug(full_metadata.as_xml())

        self.init_data_store(full_metadata)

        # TODO: account for remote acquisition by recording remote clock offsets
        # so we can map from remote timestamp to local lsl clock for comparing
        # datasets.

        ## Run loop for continous acquisition
        while self.running():
            data, timestamps = inlet.pull_chunk(
                max_samples=self.max_chunk_size)
            if timestamps:
                self.write_chunk(data, timestamps)
            time.sleep(self.sleep_seconds)

        # Pull one last chunk to account for data streaming during sleep. This
        # may result in up to (sleep_seconds * sample_rate) more records than
        # anticipated. If this is a problem we can override the `stop` method
        # and capture a timestamp, then use that timestamp to determine how
        # long we slept since receiving the call and set the max_samples
        # parameter accordingly.
        data, timestamps = inlet.pull_chunk(max_samples=self.max_chunk_size)
        if timestamps:
            self.write_chunk(data, timestamps)

        self.cleanup()


def main(path: str, seconds: int = 5, debug: bool = False):
    """Function to demo the LslRecorder. Expects LSL data streams to be already
    running."""
    if debug:
        from bcipy.helpers.system_utils import log_to_stdout
        log_to_stdout()
    recorder = LslRecorder(path)
    print(f"\nCollecting data for {seconds}s...")
    recorder.start()
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        print("Keyboard Interrupt; stopping.")
    finally:
        recorder.stop()


if __name__ == '__main__':
    import argparse
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='.')
    parser.add_argument('--seconds', default=5)
    parser.add_argument('--debug', action='store_true')
    parsed_args = parser.parse_args()
    main(path=parsed_args.path,
         seconds=int(parsed_args.seconds),
         debug=parsed_args.debug)
