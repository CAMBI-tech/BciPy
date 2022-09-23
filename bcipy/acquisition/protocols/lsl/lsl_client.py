"""DataAcquisitionClient for LabStreamingLayer data sources."""
import logging
from typing import List

from pylsl import StreamInfo, StreamInlet, local_clock, resolve_stream

from bcipy.acquisition.devices import DEFAULT_DEVICE_TYPE, DeviceSpec, IRREGULAR_RATE
from bcipy.acquisition.exceptions import InvalidClockError
from bcipy.acquisition.protocols.lsl.lsl_connector import (channel_names,
                                                           check_device)
from bcipy.acquisition.protocols.lsl.lsl_recorder import LslRecordingThread
from bcipy.acquisition.record import Record
from bcipy.helpers.clock import Clock
from bcipy.gui.viewer.ring_buffer import RingBuffer

log = logging.getLogger(__name__)


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

    def __init__(self,
                 max_buffer_len: float = 1,
                 device_spec: DeviceSpec = None,
                 save_directory: str = None,
                 raw_data_file_name: str = None):
        super().__init__()
        self.device_spec = device_spec
        self.max_buffer_len = max_buffer_len

        self.experiment_clock = None

        self.inlet = None
        self._first_sample_time = None

        self.save_directory = save_directory
        self.raw_data_file_name = raw_data_file_name

        self.recorder = None
        self.buffer = None

    def start_acquisition(self) -> bool:
        """Connect to the datasource and start acquiring data.

        Returns
        -------
        bool : False if acquisition is already in progress, otherwise True.
        """
        if self.inlet:
            return False

        content_type = self.device_spec.content_type if self.device_spec else DEFAULT_DEVICE_TYPE
        streams = resolve_stream('type', content_type)
        if not streams:
            raise Exception(
                f'LSL Stream not found for content type {content_type}')
        stream_info = streams[0]

        self.inlet = StreamInlet(stream_info, max_buflen=self.max_buffer_len)

        if self.device_spec:
            check_device(self.device_spec, self.inlet.info())
        else:
            self.device_spec = device_from_metadata(self.inlet.info())

        if self.save_directory:
            self.recorder = LslRecordingThread(stream_info,
                                               self.save_directory,
                                               self.raw_data_file_name,
                                               self.device_spec)
            self.recorder.start()

        if self.max_buffer_len and self.max_buffer_len > 0:
            self.buffer = RingBuffer(size_max=self.max_samples)
        _, self._first_sample_time = self.inlet.pull_sample()
        return True

    @property
    def first_sample_time(self) -> float:
        """Timestamp returned by the first sample. If the data is being
        recorded this value reflects the timestamp of the first recorded sample"""
        if self.recorder:
            return self.recorder.first_sample_time
        return self._first_sample_time

    def stop_acquisition(self) -> None:
        """Disconnect from the data source."""
        log.debug("Stopping Acquisition...")
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None
        if self.recorder:
            self.recorder.stop()
            self.recorder.join()

        self.buffer = None

    def __enter__(self):
        """Context manager enter method that starts data acquisition."""
        self.start_acquisition()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        """Context manager exit method to clean up resources."""
        self.stop_acquisition()

    def get_data(self,
                 start: float = None,
                 end: float = None,
                 limit: int = None) -> List[Record]:
        """Get data in time range.

        Parameters
        ----------
            start : starting timestamp (acquisition clock).
            end : end timestamp (in acquisition clock).
            limit: the max number of records that should be returned.

        Returns
        -------
            List of Records
        """
        log.debug(f"Getting data from: {start} to: {end} limit: {limit}")

        # Only data in the current buffer is available to query;
        # requests for data outside of this will fail. Buffer size is
        # set using the max_buffer_len parameter.
        data = self.get_latest_data()

        if not data:
            log.debug('No records available')
            return []

        log.debug((f'{len(data)} records available '
                   f'(From: {data[0].timestamp} To: {data[-1].timestamp})'))
        start = start or data[0].timestamp
        end = end or data[-1].timestamp
        limit = limit or -1
        assert start >= data[0].timestamp, (
            f'Start time of {start} is out of range: '
            f'({data[0].timestamp} to {data[-1].timestamp}).')

        data_slice = [
            record for record in data if start <= record.timestamp <= end
        ][0:limit]
        log.debug(f'{len(data_slice)} records returned')
        return data_slice

    @property
    def max_samples(self) -> int:
        """Maximum number of samples available at any given time."""
        if self.device_spec.sample_rate == IRREGULAR_RATE:
            return int(self.max_buffer_len)
        return int(self.max_buffer_len * self.device_spec.sample_rate)

    def get_latest_data(self) -> List[Record]:
        """Add all available samples in the inlet to the buffer.

        The number of items returned depends on the size of the configured
        max_buffer_len and the amount of data available in the inlet."""
        if not self.buffer:
            return []
        samples, timestamps = self.inlet.pull_chunk(
            max_samples=self.max_samples)

        for i, sample in enumerate(samples):
            self.buffer.append(
                Record(data=sample, timestamp=timestamps[i], rownum=None))

        return self.buffer.get()

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

    def clock_offset(self, experiment_clock: Clock = None) -> float:
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
        log.debug(f"Acquisition offset: {offset_from_stim}")
        return offset_from_stim

    def cleanup(self):
        """Perform any necessary cleanup."""


def device_from_metadata(metadata: StreamInfo) -> DeviceSpec:
    """Create a device_spec from the data stream metadata."""
    return DeviceSpec(name=metadata.name(),
                      channels=channel_names(metadata),
                      sample_rate=metadata.nominal_srate(),
                      content_type=metadata.type())
