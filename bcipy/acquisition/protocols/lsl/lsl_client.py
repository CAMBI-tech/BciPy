"""DataAcquisitionClient for LabStreamingLayer data sources."""
import logging
from multiprocessing import Queue
from typing import Dict, List, Optional, Any

import pandas as pd
from pylsl import StreamInlet, local_clock, resolve_byprop

from bcipy.acquisition.devices import IRREGULAR_RATE, DeviceSpec
from bcipy.acquisition.exceptions import InvalidClockError
from bcipy.acquisition.protocols.lsl.connect import (device_from_metadata,
                                                     resolve_device_stream)
from bcipy.acquisition.protocols.lsl.lsl_connector import check_device
from bcipy.acquisition.protocols.lsl.lsl_recorder import LslRecordingThread
from bcipy.acquisition.record import Record
from bcipy.config import MAX_PAUSE_SECONDS, SESSION_LOG_FILENAME
from bcipy.gui.viewer.ring_buffer import RingBuffer
from bcipy.helpers.clock import Clock

LSL_TIMEOUT = 5.0  # seconds

logger = logging.getLogger(SESSION_LOG_FILENAME)


def time_range(stamps: List[float],
               precision: int = 3,
               sep: str = " to ") -> str:
    """Utility for formatting a range of timestamps into a string.

    Args:
        stamps (List[float]): A list of timestamps.
        precision (int, optional): The number of decimal places for rounding timestamps.
                                   Defaults to 3.
        sep (str, optional): The separator string between the start and end timestamps.
                             Defaults to " to ".

    Returns:
        str: A string representing the range of timestamps (e.g., "1.234 to 5.678"),
             or an empty string if `stamps` is empty.
    """
    if stamps:
        return "".join([
            str(round(stamps[0], precision)), sep,
            str(round(stamps[-1], precision))
        ])
    return ""


def request_desc(start: Optional[float], end: Optional[float],
                 limit: Optional[int]) -> str:
    """Returns a descriptive string of a data request for logging purposes.

    Args:
        start (Optional[float]): The starting timestamp of the request.
        end (Optional[float]): The ending timestamp of the request.
        limit (Optional[int]): The maximum number of records requested.

    Returns:
        str: A formatted string describing the data request.
    """
    start_str = round(start, 3) if start else "None"
    end_str = round(end, 3) if end else "None"
    return f"Requesting data from: {start_str} to: {end_str} limit: {limit}"


class LslAcquisitionClient:
    """Data Acquisition Client for devices streaming data using Lab Streaming
    Layer.

    Its primary use is dynamically querying streaming data in realtime.
    If `save_directory` and `filename` parameters are provided, it uses a
    `LslRecordingThread` to persist the data.

    Args:
        max_buffer_len (float, optional): The maximum length of data to be queried.
                                          For continuously streaming data, this is the
                                          number of seconds of data to retain. For
                                          irregular data, it specifies the number of samples.
                                          When using the RSVP paradigm, `max_buffer_len`
                                          should be large enough to store data for the
                                          entire inquiry. Defaults to 1.
        device_spec (Optional[DeviceSpec], optional): The `DeviceSpec` for the device
                                                     from which to query data.
                                                     If missing, this class will attempt
                                                     to find the first EEG stream. Defaults to None.
        save_directory (Optional[str], optional): If present, persists the data to
                                                  the given location. Defaults to None.
        raw_data_file_name (Optional[str], optional): If present, uses this name
                                                      for the data file. Defaults to None.
    """

    inlet: Optional[StreamInlet] = None
    recorder: Optional[LslRecordingThread] = None
    buffer: Optional[RingBuffer] = None
    _first_sample_time: Optional[float] = None
    experiment_clock: Optional[Clock] = None

    def __init__(self,
                 max_buffer_len: float = 1,
                 device_spec: Optional[DeviceSpec] = None,
                 save_directory: Optional[str] = None,
                 raw_data_file_name: Optional[str] = None):
        super().__init__()

        self.device_spec: Optional[DeviceSpec] = device_spec
        self.max_buffer_len: float = max_buffer_len
        self.save_directory: Optional[str] = save_directory
        self.raw_data_file_name: Optional[str] = raw_data_file_name
        self._max_samples: Optional[int] = None

    @property
    def has_irregular_rate(self) -> bool:
        """Checks if the device has an irregular sampling rate.

        Returns:
            bool: True if the device's sample rate is `IRREGULAR_RATE`,
                  False otherwise.
        """
        return self.device_spec.sample_rate == IRREGULAR_RATE  # type: ignore

    @property
    def first_sample_time(self) -> Optional[float]:
        """Returns the timestamp of the first sample.

        If data is being recorded, this value reflects the timestamp of the first
        recorded sample.

        Returns:
            Optional[float]: The timestamp of the first sample, or None if not set.
        """
        return self._first_sample_time

    @property
    def max_samples(self) -> int:
        """Calculates the maximum number of samples available at any given time.

        This depends on `max_buffer_len` and the device's sample rate.

        Returns:
            int: The maximum number of samples.
        """
        if self._max_samples is None:
            if self.has_irregular_rate:
                self._max_samples = int(self.max_buffer_len)
            else:
                self._max_samples = int(self.max_buffer_len *
                                        self.device_spec.sample_rate)  # type: ignore
        return self._max_samples

    def start_acquisition(self) -> bool:
        """Connects to the data source and begins acquiring data.

        Initializes the LSL `StreamInlet` and optionally an `LslRecordingThread`
        if a `save_directory` is provided.

        Returns:
            bool: False if acquisition is already in progress, otherwise True.
        """
        if self.inlet:
            return False

        stream_info = resolve_device_stream(self.device_spec)
        self.inlet = StreamInlet(
            stream_info,
            max_buflen=MAX_PAUSE_SECONDS,
            max_chunklen=1)
        logger.info("Acquiring data from data stream:")
        logger.info(self.inlet.info().as_xml())

        if self.device_spec:
            check_device(self.device_spec, self.inlet.info())
        else:
            self.device_spec = device_from_metadata(self.inlet.info())

        if self.save_directory:
            msg_queue: Queue[float] = Queue()
            self.recorder = LslRecordingThread(
                directory=self.save_directory,
                filename=self.raw_data_file_name,
                device_spec=self.device_spec,
                queue=msg_queue)
            self.recorder.start()
            if not self.has_irregular_rate:
                logger.info("Waiting for first sample from lsl_recorder")
                self._first_sample_time = msg_queue.get(block=True,
                                                        timeout=LSL_TIMEOUT)
                logger.info(f"First sample time: {self.first_sample_time}")

        self.inlet.open_stream(timeout=LSL_TIMEOUT)
        if self.max_buffer_len and self.max_buffer_len > 0:
            self.buffer = RingBuffer(size_max=self.max_samples)
        if not self._first_sample_time:
            timeout = 0.0 if self.has_irregular_rate else LSL_TIMEOUT
            _, self._first_sample_time = self.inlet.pull_sample(
                timeout=timeout)
        return True

    def stop_acquisition(self) -> None:
        """Disconnects from the data source and cleans up resources.

        Stops the `LslRecordingThread` if active, closes the LSL `StreamInlet`,
        and clears the internal buffer.
        """
        logger.info(f"Stopping Acquisition from {self.device_spec.name} ...")  # type: ignore
        if self.recorder:
            logger.info(f"Closing  {self.device_spec.name} data recorder")  # type: ignore
            self.recorder.stop()
            self.recorder.join()
        if self.inlet:
            logger.info("Closing LSL connection")
            self.inlet.close_stream()
            self.inlet = None
            logger.info("Inlet closed")

        self.buffer = None

    def __enter__(self) -> 'LslAcquisitionClient':
        """Context manager enter method that starts data acquisition.

        Returns:
            LslAcquisitionClient: The instance of the acquisition client.
        """
        self.start_acquisition()
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        """Context manager exit method to clean up resources.

        Args:
            _exc_type (Any): The exception type, if an exception was raised.
            _exc_value (Any): The exception value, if an exception was raised.
            _traceback (Any): The traceback, if an exception was raised.
        """
        self.stop_acquisition()

    def _data_stats(self, data: List[Record]) -> Dict[str, float]:
        """Summarizes a list of records for logging and inspection.

        Args:
            data (List[Record]): A list of `Record` objects.

        Returns:
            Dict[str, float]: A dictionary containing statistics such as count,
                              total seconds, start/end timestamps, expected difference,
                              mean difference, and max difference between samples.
                              Returns an empty dict if `data` is empty.
        """
        if data:
            diffs = pd.DataFrame(data)['timestamp'].diff()
            data_start = data[0].timestamp
            data_end = data[-1].timestamp
            precision = 3
            expected_diff = 0.0 if self.has_irregular_rate else round(
                1 / self.device_spec.sample_rate, precision)  # type: ignore
            return {
                'count': len(data),
                'seconds': round(data_end - data_start, precision),
                'from': round(data_start, precision),
                'to': round(data_end, precision),
                'expected_diff': expected_diff,
                'mean_diff': round(diffs.mean(), precision),
                'max_diff': round(diffs.max(), precision)
            }
        return {}

    def get_data(self,
                 start: Optional[float] = None,
                 end: Optional[float] = None,
                 limit: Optional[int] = None) -> List[Record]:
        """Retrieves data within a specified time range from the current buffer.

        Only data currently in the buffer is available for querying.
        Requests for data outside of this range will fail.

        Args:
            start (Optional[float]): The starting timestamp (in acquisition clock).
                                     Defaults to None, which means the beginning
                                     of available data.
            end (Optional[float]): The end timestamp (in acquisition clock).
                                   Defaults to None, which means the end of
                                   available data.
            limit (Optional[int]): The maximum number of records to return.
                                   Defaults to None, which means no limit.

        Returns:
            List[Record]: A list of `Record` objects within the specified range.
                          Returns an empty list if no records are available.

        Raises:
            AssertionError: If `start` or `end` times are out of the available data range
                            for regular rate devices.
        """
        logger.info(request_desc(start, end, limit))

        data = self.get_latest_data()
        if not data:
            logger.info('No records available')
            return []

        data_start = data[0].timestamp
        data_end = data[-1].timestamp
        logger.info(f'Available data: {self._data_stats(data)}')

        if start is None:
            start = data_start
        if end is None:
            end = data_end

        if not self.has_irregular_rate:
            assert start is not None and start >= data_start, 'Start time out of range'
            assert end is not None and end <= data_end, 'End time out of range'

        data_slice = [
            record for record in data if start <= record.timestamp <= end  # type: ignore
        ][0:limit]
        logger.info(f"Filtered records: {self._data_stats(data_slice)}")

        return data_slice

    def get_latest_data(self) -> List[Record]:
        """Adds all currently available samples from the inlet to the buffer.

        The number of items returned depends on the size of the configured
        `max_buffer_len` and the amount of data available in the inlet.

        Returns:
            List[Record]: A list of the latest `Record` objects from the buffer.
                          Returns an empty list if no buffer is initialized.
        """
        if not self.buffer:
            return []

        count = self._pull_chunk()
        # Pull all the data from LSL and append it to the local buffer.
        while count == self.max_samples:
            count = self._pull_chunk()

        return self.buffer.get()

    def _pull_chunk(self) -> int:
        """Pulls a chunk of samples from LSL and records them in the buffer.

        Returns:
            int: The count of samples pulled in this operation.
        """
        logger.debug(f"\tPulling chunk (max_samples: {self.max_samples})")
        # A timeout of 0.0 gets currently available samples without blocking.
        samples, timestamps = self.inlet.pull_chunk(  # type: ignore
            timeout=0.0, max_samples=self.max_samples)
        count = len(samples)
        logger.debug(f"\t-> received {count} samples: {time_range(timestamps)}")
        for sample, stamp in zip(samples, timestamps):
            self.buffer.append(Record(sample, stamp))  # type: ignore
        return count

    def convert_time(self, experiment_clock: Clock, timestamp: float) -> float:
        """Converts a timestamp from the experiment clock to the acquisition clock.

        Used for querying the acquisition data for a time slice.

        Args:
            experiment_clock (Clock): The clock used to generate the timestamp.
            timestamp (float): The timestamp from the experiment clock.

        Returns:
            float: The corresponding timestamp for the acquisition clock.
        """

        # experiment_time = pylsl.local_clock() - offset
        return timestamp + self.clock_offset(experiment_clock)

    def get_data_seconds(self, seconds: int) -> List[Record]:
        """Returns the last 'n' seconds of data available in the buffer.

        Args:
            seconds (int): The number of seconds of data to retrieve.

        Returns:
            List[Record]: A list of `Record` objects covering the last `seconds`.

        Raises:
            AssertionError: If `seconds` exceeds `max_buffer_len`.
        """
        assert seconds <= self.max_buffer_len, f"Seconds can't exceed {self.max_buffer_len}"

        sample_count = seconds * self.device_spec.sample_rate  # type: ignore
        records = self.get_latest_data()

        start_index = 0 if len(
            records) > sample_count else len(records) - sample_count
        return records[int(start_index):]

    @property
    def is_calibrated(self) -> bool:
        """Checks whether acquisition has been calibrated (an offset calculated based on a trigger).

        Returns:
            bool: True, as this property is currently hardcoded to return True.
        """
        return True

    @is_calibrated.setter
    def is_calibrated(self, bool_val: bool) -> None:
        """Setter for the `is_calibrated` property.

        Allows the user to override the calculated value and use a 0 offset.

        Args:
            bool_val (bool): If True, forces a 0 offset; if False, forces the calculation.
                             Note: Current implementation always returns True for getter.
        """
        # This setter currently has no effect on the getter's return value.
        pass

    def clock_offset(self, experiment_clock: Optional[Clock] = None) -> float:
        """Calculates the offset in seconds from the experiment clock to the acquisition local clock.

        The experiment clock should be monotonic from experiment start time. The
        acquisition clock (`pylsl.local_clock()`) is monotonic from local machine
        start time (or since 1970-01-01 00:00). Therefore the acquisition clock
        should always be greater than experiment clock.

        Args:
            experiment_clock (Optional[Clock], optional): The experiment clock object.
                                                          Defaults to None, in which
                                                          case `self.experiment_clock` is used.

        Returns:
            float: The offset in seconds.

        Raises:
            AssertionError: If an experiment clock is not provided or available.
            InvalidClockError: If the acquisition clock is not greater than the
                               experiment clock.
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
        """Computes the number of seconds that recording started prior to the given event.

        Args:
            event_clock (Clock): Monotonic clock used to record the event time.
            event_time (float): Timestamp of the event of interest.

        Returns:
            float: Seconds between acquisition start and the event, or 0.0 if `first_sample_time` is not set.
        """
        if self.first_sample_time:
            lsl_event_time = self.convert_time(event_clock, event_time)
            return lsl_event_time - self.first_sample_time
        return 0.0

    def offset(self, first_stim_time: float) -> float:
        """Calculates the offset in seconds from the start of acquisition to the given stimulus time.

        Args:
            first_stim_time (float): LSL local clock timestamp of the first stimulus.

        Returns:
            float: The number of seconds between acquisition start and the calibration
                   event, or 0.0 if `first_stim_time` is zero or `has_irregular_rate` is True.

        Raises:
            AssertionError: If `first_sample_time` is not set and `has_irregular_rate` is False.
        """
        if not first_stim_time or self.has_irregular_rate:
            return 0.0
        assert self.first_sample_time, "Acquisition was not started."
        offset_from_stim = first_stim_time - self.first_sample_time
        logger.info(f"Acquisition offset: {offset_from_stim}")
        return offset_from_stim

    def cleanup(self) -> None:
        """Performs any necessary cleanup tasks.

        Currently, this method is a placeholder and does not perform any specific actions.
        """
        pass


def discover_device_spec(content_type: str) -> DeviceSpec:
    """Finds the first LSL stream with the given content type and creates a
    device spec from the stream's metadata.

    Args:
        content_type (str): The content type of the LSL stream to discover (e.g., "EEG", "Markers").

    Returns:
        DeviceSpec: A `DeviceSpec` object created from the metadata of the discovered stream.

    Raises:
        Exception: If an LSL stream with the specified content type is not found within `LSL_TIMEOUT`.
    """
    logger.info(f"Waiting for {content_type} data to be streamed over LSL.")
    streams = resolve_byprop('type', content_type, timeout=LSL_TIMEOUT)
    if not streams:
        raise Exception(
            f'LSL Stream not found for content type {content_type}')
    stream_info = streams[0]
    inlet = StreamInlet(stream_info)
    spec = device_from_metadata(inlet.info())
    inlet.close_stream()
    return spec
