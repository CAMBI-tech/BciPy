"""Records LSL data streams to a data store."""
import logging
import time
from pathlib import Path
from typing import List, Tuple

from pylsl import StreamInfo, StreamInlet, resolve_streams

from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.protocols.lsl.lsl_connector import channel_names, check_device
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
    - device_spec : optional DeviceSpec ; if provided channel labels will come
        from here.
    """

    def __init__(self,
                 stream_info: StreamInfo,
                 directory: str,
                 filename: str = None,
                 device_spec: DeviceSpec = None):
        super().__init__()
        self.stream_info = stream_info
        self.directory = directory
        self.device_spec = device_spec

        self.sample_count = 0
        # see: https://labstreaminglayer.readthedocs.io/info/faqs.html#chunk-sizes
        self.max_chunk_size = 1024

        # seconds to sleep between data pulls from LSL
        self.sleep_seconds = 0.2
        self.writer = None

        self.filename = filename if filename else self.default_filename()
        self.first_sample_time = None
        self.last_sample_time = None

    def default_filename(self):
        """Default filename to use if a name is not provided."""
        content_type = '_'.join(self.stream_info.type().split()).lower()
        name = '_'.join(self.stream_info.name().split()).lower()
        return f"{content_type}_data_{name}.csv"

    @property
    def recorded_seconds(self) -> float:
        """Total seconds of data recorded."""
        if self.first_sample_time and self.last_sample_time:
            return self.last_sample_time - self.first_sample_time
        return 0.0

    def _init_data_writer(self, stream_info: StreamInfo) -> None:
        """Initializes the raw data writer.

        Parameters:
        ----------
        - metadata : metadata about the data stream.
        """
        assert self.writer is None, "Data store has already been initialized."

        channels = channel_names(stream_info)
        # Use the device_spec channels labels if provided.
        if self.device_spec:
            check_device(self.device_spec, stream_info)
            channels = self.device_spec.channels

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
        self.writer = None

    def _write_chunk(self, data: List, timestamps: List) -> None:
        """Persists the data resulting from pulling a chunk from the inlet.

        Parameters
        ----------
            data : list of samples
            timestamps : list of timestamps
        """
        assert self.writer, "Writer not initialized"

        chunk = []
        for i, sample in enumerate(data):
            self.sample_count += 1
            chunk.append([self.sample_count] + sample + [timestamps[i]])
        self.writer.writerows(chunk)

    def _pull_chunk(self, inlet: StreamInlet) -> Tuple[int, float]:
        """Pull a chunk of data and persist. Updates first_sample_time,
        last_sample_time, and sample_count.

        Parameters
        ----------
            inlet : stream inlet from which to pull

        Returns
        -------
            number of samples pulled
        """
        # A timeout of 0.0 does not block and only gets samples immediately
        # available.
        data, timestamps = inlet.pull_chunk(timeout=0.0,
                                            max_samples=self.max_chunk_size)
        if timestamps:
            if not self.first_sample_time:
                self.first_sample_time = timestamps[0]
            self.last_sample_time = timestamps[-1]
            self._write_chunk(data, timestamps)
        return len(timestamps)

    def _reset(self) -> None:
        """Reset state"""
        self.sample_count = 0
        self.first_sample_time = None
        self.last_sample_time = None

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

        self._reset()
        self._init_data_writer(full_metadata)

        # TODO: account for remote acquisition by recording remote clock offsets
        # so we can map from remote timestamp to local lsl clock for comparing
        # datasets.

        # Run loop for continuous acquisition
        while self.running():
            self._pull_chunk(inlet)
            time.sleep(self.sleep_seconds)

        # Pull any remaining samples up to the current time.
        log.debug("Pulling remaining samples")
        record_count = self._pull_chunk(inlet)
        while record_count == self.max_chunk_size:
            record_count = self._pull_chunk(inlet)

        log.info(f"Ending data stream recording for {self.stream_info.name()}")
        log.info(f"Total recorded seconds: {self.recorded_seconds}")
        log.info(f"Total recorded samples: {self.sample_count}")
        inlet.close_stream()
        self._cleanup()


def main(path: str, seconds: int = 5, debug: bool = False):
    """Function to demo the LslRecorder. Expects LSL data streams to be already
    running."""
    if debug:
        # pylint: disable=import-outside-toplevel
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
