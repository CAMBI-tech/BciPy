"""Records LSL data streams to a data store."""
import logging
import time
from multiprocessing import Queue
from pathlib import Path
from typing import List, Optional

from pylsl import StreamInfo, StreamInlet, resolve_streams

from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.protocols.lsl.connect import (device_from_metadata,
                                                     resolve_device_stream)
from bcipy.acquisition.protocols.lsl.lsl_connector import (channel_names,
                                                           check_device)
from bcipy.acquisition.util import StoppableProcess
from bcipy.config import SESSION_LOG_FILENAME
from bcipy.helpers.raw_data import RawDataWriter

log = logging.getLogger(SESSION_LOG_FILENAME)


class LslRecorder:
    """Records LSL data to a datastore. Resolves streams when started.

    Parameters:
    -----------
    - path : location to store the recordings
    - filenames : optional dict mapping device type to its raw data filename.
    Devices without an entry will use a naming convention.
    """

    streams: List['LslRecordingThread'] = None

    def __init__(self, path: str, filenames: Optional[dict] = None) -> None:
        super().__init__()
        self.path = path
        self.filenames = filenames or {}

    def start(self) -> None:
        """Start recording all streams currently on the network."""

        if not self.streams:
            log.info("Recording data")
            # create a thread for each.
            self.streams = [
                LslRecordingThread(device_spec=device_from_metadata(StreamInlet(stream).info()),
                                   directory=self.path,
                                   filename=self.filenames.get(stream.type(), None))
                for stream in resolve_streams()
            ]

            # Validate that streams have unique names for their type
            stream_names = [stream.filename for stream in self.streams]
            if len(stream_names) != len(set(stream_names)):
                raise Exception("Data stream names are not unique")
            for stream in self.streams:
                stream.start()

    def stop(self, wait: bool = False) -> None:
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


class LslRecordingThread(StoppableProcess):
    """Records data for the given LabStreamingLayer (LSL) data stream.

    Parameters:
    ----------
    - device_spec : DeviceSpec ; specifies the device from which to record.
    - directory : location to store the recording
    - filename : optional, name of the data file.
    - queue : optional multiprocessing queue; if provided the first_sample_time
        will be written here when available.
    """

    writer: RawDataWriter = None

    def __init__(self,
                 device_spec: DeviceSpec,
                 directory: Optional[str] = '.',
                 filename: Optional[str] = None,
                 queue: Optional[Queue] = None) -> None:
        super().__init__()

        self.directory = directory
        self.device_spec = device_spec
        self.queue = queue

        self.sample_count = 0
        # see: https://labstreaminglayer.readthedocs.io/info/faqs.html#chunk-sizes
        self.max_chunk_size = 1024

        # seconds to sleep between data pulls from LSL
        self.sleep_seconds = 0.2

        self.filename = filename if filename else self.default_filename()
        self.first_sample_time = None
        self.last_sample_time = None

    def default_filename(self):
        """Default filename to use if a name is not provided."""
        content_type = '_'.join(self.device_spec.content_type.split()).lower()
        name = '_'.join(self.device_spec.name.split()).lower()
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

        path = str(Path(self.directory, self.filename))
        log.info(f"Writing data to {path}")
        self.writer = RawDataWriter(
            path,
            daq_type=stream_info.name(),
            sample_rate=stream_info.nominal_srate(),
            columns=['timestamp'] + channels + ['lsl_timestamp'])
        self.writer.__enter__()

    def _cleanup(self) -> None:
        """Performs cleanup tasks."""
        if self.writer:
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

    def _pull_chunk(self, inlet: StreamInlet) -> int:
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
        if timestamps and data:
            if not self.first_sample_time:
                self.first_sample_time = timestamps[0]
                if self.queue:
                    self.queue.put(self.first_sample_time, timeout=2.0)
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
        # Note that stream_info does not have the channel names.
        stream_info = resolve_device_stream(self.device_spec)
        inlet = StreamInlet(stream_info, max_chunklen=1)
        full_metadata = inlet.info()

        log.info("Recording data from data stream:")
        log.info(full_metadata.as_xml())

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
        log.info("Pulling remaining samples")
        record_count = self._pull_chunk(inlet)
        while record_count == self.max_chunk_size:
            record_count = self._pull_chunk(inlet)

        log.info(f"Ending data stream recording for {stream_info.name()}")
        log.info(f"Total recorded seconds: {self.recorded_seconds}")
        log.info(f"Total recorded samples: {self.sample_count}")
        inlet.close_stream()
        inlet = None
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
