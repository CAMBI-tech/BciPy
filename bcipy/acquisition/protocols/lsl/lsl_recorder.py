"""Records LSL data streams to a data store."""
import logging
import time
from multiprocessing import Queue
from pathlib import Path
from typing import List, Optional, Any

from pylsl import StreamInfo, StreamInlet, resolve_streams

from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.protocols.lsl.connect import (device_from_metadata,
                                                     resolve_device_stream)
from bcipy.acquisition.protocols.lsl.lsl_connector import (channel_names,
                                                           check_device)
from bcipy.acquisition.util import StoppableProcess
from bcipy.config import SESSION_LOG_FILENAME
from bcipy.core.raw_data import RawDataWriter
from bcipy.helpers.utils import log_to_stdout

log = logging.getLogger(SESSION_LOG_FILENAME)


class LslRecorder:
    """Records LSL data to a datastore. Resolves streams when started.

    Args:
        path (str): Location to store the recordings.
        filenames (Optional[dict], optional): Optional dictionary mapping device
                                             type to its raw data filename.
                                             Devices without an entry will use a
                                             naming convention. Defaults to None.
    """

    streams: Optional[List['LslRecordingThread']] = None

    def __init__(self, path: str, filenames: Optional[dict] = None) -> None:
        super().__init__()
        self.path: str = path
        self.filenames: dict = filenames or {}

    def start(self) -> None:
        """Starts recording all LSL streams currently on the network.

        This method creates an `LslRecordingThread` for each discovered stream
        and starts them. It also validates that stream names are unique.

        Raises:
            Exception: If data stream names are not unique.
        """

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
        """Stops recording for all active streams.

        Args:
            wait (bool, optional): If True, waits for all recording threads
                                   to stop before returning. Defaults to False.
        """
        if self.streams:
            for stream in self.streams:
                stream.stop()
                if wait:
                    stream.join()
            self.streams = None


class LslRecordingThread(StoppableProcess):
    """Records data for the given LabStreamingLayer (LSL) data stream.

    This class extends `StoppableProcess` to run recording in a separate process.

    Args:
        device_spec (DeviceSpec): Specifies the device from which to record.
        directory (Optional[str], optional): Location to store the recording.
                                            Defaults to '.'.
        filename (Optional[str], optional): Optional name of the data file.
                                            If None, a default filename based
                                            on device properties will be used.
                                            Defaults to None.
        queue (Optional[Queue], optional): Optional multiprocessing queue.
                                           If provided, the `first_sample_time`
                                           will be written to this queue when available.
                                           Defaults to None.
    """

    writer: Optional[RawDataWriter] = None

    def __init__(self,
                 device_spec: DeviceSpec,
                 directory: Optional[str] = '.',
                 filename: Optional[str] = None,
                 queue: Optional[Queue] = None) -> None:
        super().__init__()

        self.directory: Optional[str] = directory
        self.device_spec: DeviceSpec = device_spec
        self.queue: Optional[Queue] = queue

        self.sample_count: int = 0
        # see: https://labstreaminglayer.readthedocs.io/info/faqs.html#chunk-sizes
        self.max_chunk_size: int = 1024

        # seconds to sleep between data pulls from LSL
        self.sleep_seconds: float = 0.2

        self.filename: str = filename if filename else self.default_filename()
        self.first_sample_time: Optional[float] = None
        self.last_sample_time: Optional[float] = None

    def default_filename(self) -> str:
        """Generates a default filename to use if a name is not provided.

        The filename is based on the device's content type and name.

        Returns:
            str: The generated default filename (e.g., "eeg_data_dsi_24.csv").
        """
        content_type = '_'.join(self.device_spec.content_type.split()).lower()
        name = '_'.join(self.device_spec.name.split()).lower()
        return f"{content_type}_data_{name}.csv"

    @property
    def recorded_seconds(self) -> float:
        """Calculates the total seconds of data recorded.

        Returns:
            float: The duration of recorded data in seconds, or 0.0 if recording
                   hasn't started or completed.
        """
        if self.first_sample_time and self.last_sample_time:
            return self.last_sample_time - self.first_sample_time
        return 0.0

    def _init_data_writer(self, stream_info: StreamInfo) -> None:
        """Initializes the raw data writer.

        Args:
            stream_info (StreamInfo): Metadata about the data stream.

        Raises:
            AssertionError: If the data writer has already been initialized.
        """
        assert self.writer is None, "Data store has already been initialized."

        channels = channel_names(stream_info)
        # Use the device_spec channels labels if provided.
        if self.device_spec:
            check_device(self.device_spec, stream_info)
            channels = self.device_spec.channels

        path = str(Path(self.directory, self.filename))  # type: ignore
        log.info(f"Writing data to {path}")
        self.writer = RawDataWriter(
            path,
            daq_type=stream_info.name(),
            sample_rate=stream_info.nominal_srate(),
            columns=['timestamp'] + channels + ['lsl_timestamp'])
        self.writer.__enter__()

    def _cleanup(self) -> None:
        """Performs cleanup tasks for the data writer.

        Closes the `RawDataWriter` if it was initialized.
        """
        if self.writer:
            self.writer.__exit__()
            self.writer = None

    def _write_chunk(self, data: List[List[Any]], timestamps: List[float]) -> None:
        """Persists a chunk of data pulled from the LSL inlet.

        Args:
            data (List[List[Any]]): A list of samples, where each sample is a list of channel values.
            timestamps (List[float]): A list of timestamps corresponding to each sample.
        """
        assert self.writer, "Writer not initialized"
        chunk: List[List[Any]] = []
        for i, sample in enumerate(data):
            self.sample_count += 1
            chunk.append([self.sample_count] + sample + [timestamps[i]])
        self.writer.writerows(chunk)

    def _pull_chunk(self, inlet: StreamInlet) -> int:
        """Pulls a chunk of data from the `StreamInlet` and persists it.

        Updates `first_sample_time`, `last_sample_time`, and `sample_count`.

        Args:
            inlet (StreamInlet): The LSL `StreamInlet` from which to pull data.

        Returns:
            int: The number of samples pulled in this operation.
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
        """Resets the internal state of the recorder.

        This includes resetting the sample count and clearing the first and last
        sample timestamps.
        """
        self.sample_count = 0
        self.first_sample_time = None
        self.last_sample_time = None

    # @override
    def run(self) -> None:
        """Process startup and main recording loop.

        Connects to the device, continuously reads chunks of data at the
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


def main(path: str, seconds: int = 5, debug: bool = False) -> None:
    """Demonstrates the `LslRecorder` functionality.

    This function initializes an `LslRecorder` and records data for a specified
    duration. It expects LSL data streams to be already running.

    Args:
        path (str): The directory path to save the recorded data.
        seconds (int, optional): The duration in seconds to record data.
                                Defaults to 5.
        debug (bool, optional): If True, enables logging to stdout for debugging.
                                Defaults to False.
    """
    if debug:
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
