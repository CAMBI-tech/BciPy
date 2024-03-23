"""DataAcquisitionClient for LabStreamingLayer data sources."""
import logging
from multiprocessing import Queue
from typing import Dict, List, Optional

import pandas as pd
from pylsl import StreamInlet, local_clock, resolve_byprop

from bcipy.acquisition.devices import IRREGULAR_RATE, DeviceSpec
from bcipy.acquisition.exceptions import InvalidClockError
from bcipy.acquisition.protocols.lsl.connect import (device_from_metadata,
                                                     resolve_device_stream)
from bcipy.acquisition.protocols.lsl.lsl_connector import check_device
from bcipy.acquisition.protocols.lsl.lsl_recorder import LslRecordingThread
from bcipy.acquisition.record import Record
from bcipy.config import MAX_PAUSE_SECONDS
from bcipy.gui.viewer.ring_buffer import RingBuffer
from bcipy.helpers.clock import Clock

log = logging.getLogger(__name__)


def time_range(stamps: List[float],
               precision: int = 3,
               sep: str = " to ") -> str:
    """Utility for printing a range of timestamps"""
    if stamps:
        return "".join([
            str(round(stamps[0], precision)), sep,
            str(round(stamps[-1], precision))
        ])
    return ""


def request_desc(start: Optional[float], end: Optional[float],
                 limit: Optional[int]):
    """Returns a description of the request which can be logged."""
    start_str = round(start, 3) if start else "None"
    end_str = round(end, 3) if end else "None"
    return f"Requesting data from: {start_str} to: {end_str} limit: {limit}"


class LslAcquisitionClient:
    """Data Acquisition Client for devices streaming data using Lab Streaming
    Layer. Its primary use is dynamically querying streaming data in realtime,
    however, if the save_directory and filename parameters are provided it uses
    a LslRecordingThread to persist the data.

    Parameters
    ----------
        max_buffer_len: the maximum length of data to be queried. For continuously
            streaming data this is the number of seconds of data to retain. For
            irregular data, specify the number of samples. When using the RSVP
            paradigm, the max_buffer_len should be large enough to store data for
            the entire inquiry.
        device_spec: spec for the device from which to query data; if missing,
            this class will attempt to find the first EEG stream.
        save_directory: if present, persists the data to the given location.
        raw_data_file_name: if present, uses this name for the data file.
    """

    inlet: StreamInlet = None
    recorder: LslRecordingThread = None
    buffer: RingBuffer = None
    _first_sample_time: float = None
    experiment_clock: Clock = None

    def __init__(self,
                 max_buffer_len: float = 1,
                 device_spec: Optional[DeviceSpec] = None,
                 save_directory: Optional[str] = None,
                 raw_data_file_name: Optional[str] = None):
        super().__init__()

        self.device_spec = device_spec
        self.max_buffer_len = max_buffer_len
        self.save_directory = save_directory
        self.raw_data_file_name = raw_data_file_name
        self._max_samples = None

    @property
    def first_sample_time(self) -> float:
        """Timestamp returned by the first sample. If the data is being
        recorded this value reflects the timestamp of the first recorded sample"""
        return self._first_sample_time

    @property
    def max_samples(self) -> int:
        """Maximum number of samples available at any given time."""
        if self._max_samples is None:
            if self.device_spec.sample_rate == IRREGULAR_RATE:
                self._max_samples = int(self.max_buffer_len)
            else:
                self._max_samples = int(self.max_buffer_len *
                                        self.device_spec.sample_rate)
        return self._max_samples

    def start_acquisition(self) -> bool:
        """Connect to the datasource and start acquiring data.

        Returns
        -------
        bool : False if acquisition is already in progress, otherwise True.
        """
        if self.inlet:
            return False

        stream_info = resolve_device_stream(self.device_spec)
        self.inlet = StreamInlet(
            stream_info,
            max_buflen=MAX_PAUSE_SECONDS + 5,
            max_chunklen=1)
        log.info("Acquiring data from data stream:")
        log.info(self.inlet.info().as_xml())

        if self.device_spec:
            check_device(self.device_spec, self.inlet.info())
        else:
            self.device_spec = device_from_metadata(self.inlet.info())

        if self.save_directory:
            msg_queue = Queue()
            self.recorder = LslRecordingThread(
                directory=self.save_directory,
                filename=self.raw_data_file_name,
                device_spec=self.device_spec,
                queue=msg_queue)
            self.recorder.start()
            log.info("Waiting for first sample from lsl_recorder")
            self._first_sample_time = msg_queue.get(block=True, timeout=5.0)
            log.info(f"First sample time: {self.first_sample_time}")

        self.inlet.open_stream(timeout=5.0)
        if self.max_buffer_len and self.max_buffer_len > 0:
            self.buffer = RingBuffer(size_max=self.max_samples)
        if not self._first_sample_time:
            _, self._first_sample_time = self.inlet.pull_sample()
        return True

    def stop_acquisition(self) -> None:
        """Disconnect from the data source."""
        log.info(f"Stopping Acquisition from {self.device_spec.name} ...")
        if self.recorder:
            log.info(f"Closing  {self.device_spec.name} data recorder")
            self.recorder.stop()
            self.recorder.join()
        if self.inlet:
            log.info("Closing LSL connection")
            self.inlet.close_stream()
            self.inlet = None
            log.info("Inlet closed")

        self.buffer = None

    def __enter__(self):
        """Context manager enter method that starts data acquisition."""
        self.start_acquisition()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        """Context manager exit method to clean up resources."""
        self.stop_acquisition()

    def _data_stats(self, data: List[Record]) -> Dict[str, float]:
        """Summarize a list of records for logging and inspection."""
        if data:
            diffs = pd.DataFrame(data)['timestamp'].diff()
            data_start = data[0].timestamp
            data_end = data[-1].timestamp
            precision = 3
            return {
                'count': len(data),
                'seconds': round(data_end - data_start, precision),
                'from': round(data_start, precision),
                'to': round(data_end, precision),
                'expected_diff': round(1 / self.device_spec.sample_rate,
                                       precision),
                'mean_diff': round(diffs.mean(), precision),
                'max_diff': round(diffs.max(), precision)
            }
        return {}

    def get_data(self,
                 start: Optional[float] = None,
                 end: Optional[float] = None,
                 limit: Optional[int] = None) -> List[Record]:
        """Get data in time range.

        Only data in the current buffer is available to query;
        requests for data outside of this will fail.

        Parameters
        ----------
            start : starting timestamp (acquisition clock).
            end : end timestamp (in acquisition clock).
            limit: the max number of records that should be returned.

        Returns
        -------
            List of Records
        """
        log.info(request_desc(start, end, limit))

        data = self.get_latest_data()
        if not data:
            log.info('No records available')
            return []

        data_start = data[0].timestamp
        data_end = data[-1].timestamp
        log.info(f'Available data: {self._data_stats(data)}')

        if start is None:
            start = data_start
        if end is None:
            end = data_end

        assert start >= data_start, 'Start time out of range'
        assert end <= data_end, 'End time out of range'

        data_slice = [
            record for record in data if start <= record.timestamp <= end
        ][0:limit]
        log.info(f"Filtered records: {self._data_stats(data_slice)}")

        return data_slice

    def get_latest_data(self) -> List[Record]:
        """Add all available samples in the inlet to the buffer.

        The number of items returned depends on the size of the configured
        max_buffer_len and the amount of data available in the inlet."""
        if not self.buffer:
            return []

        count = self._pull_chunk()
        # Pull all the data from LSL and append it to the local buffer.
        while count == self.max_samples:
            count = self._pull_chunk()

        return self.buffer.get()

    def _pull_chunk(self) -> int:
        """Pull a chunk of samples from LSL and record in the buffer.
        Returns the count of samples pulled.
        """
        log.debug(f"\tPulling chunk (max_samples: {self.max_samples})")
        # A timeout of 0.0 gets currently available samples without blocking.
        samples, timestamps = self.inlet.pull_chunk(
            timeout=0.0, max_samples=self.max_samples)
        count = len(samples)
        log.debug(f"\t-> received {count} samples: {time_range(timestamps)}")
        for sample, stamp in zip(samples, timestamps):
            self.buffer.append(Record(sample, stamp))
        return count

    def convert_time(self, experiment_clock: Clock, timestamp: float) -> float:
        """
        Convert a timestamp from the experiment clock to the acquisition clock.
        Used for querying the acquisition data for a time slice.

        Parameters:
        ----------
        - experiment_clock : clock used to generate the timestamp
        - timestamp : timestamp from the experiment clock

        Returns:
        --------
            corresponding timestamp for the acquistion clock
        """

        # experiment_time = pylsl.local_clock() - offset
        return timestamp + self.clock_offset(experiment_clock)

    def get_data_seconds(self, seconds: int) -> List[Record]:
        """Returns the last n second of data"""
        assert seconds <= self.max_buffer_len, f"Seconds can't exceed {self.max_buffer_len}"

        sample_count = seconds * self.device_spec.sample_rate
        records = self.get_latest_data()

        start_index = 0 if len(
            records) > sample_count else len(records) - sample_count
        return records[start_index:]

    @property
    def is_calibrated(self):
        """Returns boolean indicating whether or not acquisition has been
        calibrated (an offset calculated based on a trigger)."""
        return True

    @is_calibrated.setter
    def is_calibrated(self, bool_val):
        """Setter for the is_calibrated property that allows the user to
        override the calculated value and use a 0 offset.

        Parameters
        ----------
        - bool_val : boolean
        if True, uses a 0 offset; if False forces the calculation.
        """

    def clock_offset(self, experiment_clock: Optional[Clock] = None) -> float:
        """
        Offset in seconds from the experiment clock to the acquisition local clock.

        The experiment clock should be monotonic from experiment start time.
        The acquisition clock (pylsl.local_clock()) is monotonic from local
        machine start time (or since 1970-01-01 00:00). Therefore the acquisition
        clock should always be greater than experiment clock. An exception is
        raised if this doesn't hold.

        See https://labstreaminglayer.readthedocs.io/info/faqs.html#lsl-local-clock
        """
        clock = experiment_clock or self.experiment_clock
        assert clock, "An experiment clock must be provided"

        diff = local_clock() - clock.getTime()
        if diff < 0:
            raise InvalidClockError(
                "The acquisition clock should always be greater than experiment clock"
            )
        return diff

    def event_offset(self, event_clock: Clock, event_time: float) -> float:
        """Compute number of seconds that recording started prior to the given
        event.

        Parameters
        ----------
        - event_clock : monotonic clock used to record the event time.
        - event_time : timestamp of the event of interest.

        Returns
        -------
        Seconds between acquisition start and the event.
        """
        if self.first_sample_time:
            lsl_event_time = self.convert_time(event_clock, event_time)
            return lsl_event_time - self.first_sample_time
        return 0.0

    def offset(self, first_stim_time: float) -> float:
        """Offset in seconds from the start of acquisition to the given stim
        time.

        Parameters
        ----------
        - first_stim_time : LSL local clock timestamp of the first stimulus.

        Returns
        -------
        The number of seconds between acquisition start and the calibration
        event, or 0.0 .
        """

        if not first_stim_time:
            return 0.0
        assert self.first_sample_time, "Acquisition was not started."
        offset_from_stim = first_stim_time - self.first_sample_time
        log.info(f"Acquisition offset: {offset_from_stim}")
        return offset_from_stim

    def cleanup(self):
        """Perform any necessary cleanup."""


def discover_device_spec(content_type: str) -> DeviceSpec:
    """Finds the first LSL stream with the given content type and creates a
    device spec from the stream's metadata."""
    log.info(f"Waiting for {content_type} data to be streamed over LSL.")
    streams = resolve_byprop('type', content_type, timeout=5.0)
    if not streams:
        raise Exception(
            f'LSL Stream not found for content type {content_type}')
    stream_info = streams[0]
    inlet = StreamInlet(stream_info)
    spec = device_from_metadata(inlet.info())
    inlet.close_stream()
    return spec
