"""Records LSL data streams to a data store."""
import csv
import logging
import time
from pathlib import Path
from typing import List
from pylsl import StreamInfo, StreamInlet, resolve_streams

from bcipy.acquisition.util import StoppableThread
from bcipy.acquisition.protocols.lsl.lsl_connector import channel_names

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

            # TODO: validate that streams have unique names for their type
            for stream in self.streams:
                stream.start()

    def stop(self):
        """Stop recording"""
        for stream in self.streams:
            stream.stop()
        self.streams = None


class LslRecordingThread(StoppableThread):
    """Records data for the given LabStreamingLayer data stream.

    Parameters:
    ----------
        stream - information about the stream of interest
        path - location to store the recording
    """

    def __init__(self, stream_info: StreamInfo, path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_info = stream_info
        self.path = path

        self.sample_count = 0
        self.max_chunk_size = 1024  # default value

        self.sleep_seconds = 0.2
        self.file = None
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
        filename = self.filename
        log.debug(f"Writing data to {filename}")
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file, delimiter=',')

        # write header
        self.writer.writerow(['daq_type', self.stream_info.name()])
        self.writer.writerow(['sample_rate', self.stream_info.nominal_srate()])
        self.writer.writerow(['timestamp'] + channel_names(metadata) +
                             ['lsl_timestamp'])

    def cleanup(self):
        """Performs cleanup tasks."""
        assert self.file, "File not initialized"
        self.file.close()

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
    recorder = LslRecorder(path=args.path)
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='.')
    parser.add_argument('--seconds', default=5)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(path=args.path, seconds=int(args.seconds), debug=args.debug)
