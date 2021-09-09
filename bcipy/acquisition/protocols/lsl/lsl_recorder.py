"""Records LSL data streams to a data store."""
import logging
import time
from pathlib import Path
from typing import List

from pylsl import StreamInfo, StreamInlet, resolve_streams

from bcipy.acquisition.protocols.lsl.lsl_connector import channel_names
from bcipy.acquisition.util import StoppableThread
from bcipy.helpers.raw_data import RawDataWriter

log = logging.getLogger(__name__)


class LslRecorder:
    """Records LSL data to a datastore. Resolves streams when started.

    Parameters:
    -----------
    - path : location to store the recordings
    - filenames : optional dict mapping device type to its raw data filename.
    Devices without an entry will use a naming convention.
    """

    def __init__(self, path: str, filenames: dict = None):
        super().__init__()
        self.path = path
        self.streams = None
        self.filenames = filenames or {}

    def start(self):
        """Start recording all streams currently on the network."""

        if not self.streams:
            log.debug("Recording data")
            # create a thread for each.
            self.streams = [
                LslRecordingThread(stream, self.path,
                                   self.filenames.get(stream.type(), None))
                for stream in resolve_streams()
            ]

            # Validate that streams have unique names for their type
            stream_names = [stream.filename for stream in self.streams]
            if len(stream_names) != len(set(stream_names)):
                raise Exception("Data stream names are not unique")
            for stream in self.streams:
                stream.start()

    def stop(self, wait: bool = False):
        """Stop recording.

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
    """Records data for the given LabStreamingLayer (LSL) data stream.

    Parameters:
    ----------
    - stream : information about the stream of interest
    - directory : location to store the recording
    - filename : optional, name of the data file.
    """

    def __init__(self,
                 stream_info: StreamInfo,
                 directory: str,
                 filename: str = None):
        super().__init__()
        self.stream_info = stream_info
        self.directory = directory

        self.sample_count = 0
        self.max_chunk_size = 1024

        # seconds to sleep between data pulls from LSL
        self.sleep_seconds = 0.2
        self.writer = None

        self.filename = filename if filename else self.default_filename()
        self.first_sample_time = None

    def default_filename(self):
        """Default filename to use if a name is not provided."""
        content_type = '_'.join(self.stream_info.type().split()).lower()
        name = '_'.join(self.stream_info.name().split()).lower()
        return f"{content_type}_data_{name}.csv"

    def _init_data_writer(self, channels: List[str]) -> None:
        """Initializes the raw data writer.

        Parameters:
        ----------
        - channels : list of channel names provided for each sample.
        """
        assert self.writer is None, "Data store has already been initialized."
        path = Path(self.directory, self.filename)
        log.debug(f"Writing data to {path}")
        self.writer = RawDataWriter(
            path,
            daq_type=self.stream_info.name(),
            sample_rate=self.stream_info.nominal_srate(),
            columns=['timestamp'] + channels + ['lsl_timestamp'])
        self.writer.__enter__()

    def _cleanup(self) -> None:
        """Performs cleanup tasks."""
        assert self.writer, "Writer not initialized"
        self.writer.__exit__()

    def _write_chunk(self, data: List, timestamps: List) -> None:
        """Persists the data resulting from pulling a chunk from the inlet.

        Parameters
        ----------
        - data : list of samples
        - timestamps : list of timestamps
        """
        assert self.writer, "Writer not initialized"

        chunk = []
        for i, sample in enumerate(data):
            self.sample_count += 1
            chunk.append([self.sample_count] + sample + [timestamps[i]])
        self.writer.writerows(chunk)

    # @override
    def run(self):
        """Process startup. Connects to the device, reads chunks of data at the
        given interval, and persists the results. This happens continuously
        until the `stop()` method is called.
        """
        # Note that self.stream_info does not have the channel names.
        inlet = StreamInlet(self.stream_info)
        full_metadata = inlet.info()

        log.debug("Acquiring data from data stream:")
        log.debug(full_metadata.as_xml())

        self._init_data_writer(channel_names(full_metadata))

        # TODO: account for remote acquisition by recording remote clock offsets
        # so we can map from remote timestamp to local lsl clock for comparing
        # datasets.

        latest_sample_time = 0
        # Run loop for continous acquisition
        while self.running():
            data, timestamps = inlet.pull_chunk(
                max_samples=self.max_chunk_size)
            if timestamps:
                self._write_chunk(data, timestamps)
                if not self.first_sample_time:
                    self.first_sample_time = timestamps[0]
                latest_sample_time = timestamps[-1]
            time.sleep(self.sleep_seconds)

        # Pull one last chunk to account for data streaming during sleep. This
        # may result in up to (sleep_seconds * sample_rate) more records than
        # anticipated. If this is a problem we can override the `stop` method
        # and capture a timestamp, then use that timestamp to determine how
        # long we slept since receiving the call and set the max_samples
        # parameter accordingly.
        # TODO: do we need to use max_chunk_size here
        data, timestamps = inlet.pull_chunk(max_samples=self.max_chunk_size)
        if timestamps:
            self._write_chunk(data, timestamps)
            latest_sample_time = timestamps[-1]

        log.info(f"Ending data stream recording for {self.stream_info.name()}")
        log.info(f"Total recorded seconds: {latest_sample_time - self.first_sample_time}")
        log.info(f"Total recorded samples: {self.sample_count}")
        inlet.close_stream()
        self._cleanup()

# pylint: disable=import-outside-toplevel


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
