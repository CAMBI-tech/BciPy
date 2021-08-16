"""DataAcquisitionClient for LabStreamingLayer data sources."""
import logging
from typing import List, Tuple

from psychopy.core import Clock
from pylsl import StreamInfo, StreamInlet, local_clock, resolve_stream

from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.device_info import DeviceInfo
from bcipy.acquisition.devices import DEFAULT_DEVICE_TYPE, DeviceSpec
from bcipy.acquisition.errors import InvalidClockError
from bcipy.acquisition.marker_writer import LslMarkerWriter, NullMarkerWriter, MarkerWriter
from bcipy.acquisition.protocols.lsl.lsl_connector import (channel_names,
                                                           check_device)
from bcipy.acquisition.protocols.lsl.lsl_recorder import LslRecorder
from bcipy.acquisition.record import Record

log = logging.getLogger(__name__)


def range_evaluator(start: float = None, end: float = None):
    """Returns a function that can evaluate if a value is within the given range.

    Parameters:
    -----------
    - start : optional start of the range
    - end : optional end of the range"""
    if start and end:
        return lambda value: start <= value <= end
    if start and not end:
        return lambda value: value >= start
    if not start and end:
        return lambda value: value <= end
    # Both missing; anything goes.
    return lambda record: True


class LslAcquisitionClient:
    """Data Acquisition Client for devices streaming data using Lab Streaming
    Layer. Its primary use is dynamically querying streaming data in realtime.

    TODO: support multiple devices.

    Parameters:
    -----------
    - max_buflen : the maximum length, in seconds, of data to be queried.
    For the RSVP paradigm, this should be calculated based on the total inquiry
    length.
    - device_spec : spec for the device from which to query data; if
    missing, this class will attempt to find the first EEG stream.
    - append_timestamps : if True appends the LSL timestamp as an additional
    column in any queried data.
    - save_directory : if present, uses an LslRecorder to persist the data to
    the given location.
    - raw_data_file_name : if present, uses this name for the EEG data.
    - use_marker_writer : if present, initializes a marker writer when acquisition begins.
    """

    def __init__(self,
                 max_buflen: int,
                 device_spec: DeviceSpec = None,
                 save_directory: str = None,
                 raw_data_file_name: str = None,
                 use_marker_writer: bool = False):
        super().__init__()
        if device_spec:
            ok_device = ConnectionMethod.LSL in device_spec.connection_methods
            assert ok_device, "Only LSL devices allowed."
            assert device_spec.sample_rate > 0, "Marker streams may be recorded but not queried."
        self.device_spec = device_spec
        self.max_buflen = max_buflen

        self.experiment_clock = None

        self.inlet = None
        self.first_sample = None
        self.use_marker_writer = use_marker_writer
        self.marker_writer = None

        self.recorder = None
        if save_directory:
            self.recorder = LslRecorder(path=save_directory,
                                        filenames={'EEG': raw_data_file_name})

    def _init_marker_writer(self) -> MarkerWriter:
        """Initialize the marker writer if needed."""
        if not self.marker_writer:
            self.marker_writer = LslMarkerWriter(
            ) if self.use_marker_writer else NullMarkerWriter()

        return self.marker_writer

    def start_acquisition(self) -> None:
        """Connect to the datasource and start acquiring data."""
        self._init_marker_writer()

        if self.recorder:
            self.recorder.start()

        # TODO: Should we resolve all streams and query by name?
        if self._connect_to_query_stream():
            self.first_sample = self.inlet.pull_sample()

    def _connect_to_query_stream(self) -> bool:
        """Initialize a stream inlet to use for querying data.

        PostConditions: The inlet and device_spec properties are set.
        """
        if self.inlet:
            # Acquisition is already in progress
            return False
        if self.device_spec:
            self.inlet = device_stream(self.device_spec, self.max_buflen)
        else:
            self.inlet, self.device_spec = default_stream(self.max_buflen)
        return True

    def stop_acquisition(self) -> None:
        """Disconnect from the data source."""
        if self.recorder:
            self.recorder.stop()
        self.inlet = None

        if self.marker_writer:
            self.marker_writer.cleanup()

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
                 device: DeviceSpec = None) -> List[Record]:
        """Get data in time range.

        Parameters
        ----------
        - start : number, optional start of time slice; units are those of the
        acquisition clock.
        - end : float, optional end of time slice; units are those of the
        acquisition clock.

        Returns
        -------
        list of Records
        """
        log.debug(f"Getting data from {start} to {end}")

        # TODO: use device to get data from the correct stream
        # TODO: for remote acquisition sources we need to account for sample timestamp offset
        # from local_clock(). Shouldn't matter if everything is on the same machine.

        within_time_window = range_evaluator(start, end)
        return [
            record for record in self.get_latest_data()
            if within_time_window(record.timestamp)
        ]

    def get_data_for_clock(self,
                           start_time: float,
                           end_time: float,
                           experiment_clock: Clock = None,
                           calib_time: float = None,
                           device: DeviceSpec = None) -> List[Record]:
        """Queries the data stream, using start and end values relative to a
        clock different than the acquisition clock.

        Parameters
        ----------
        - start_time : float, optional
        start of time slice; units are those of the experiment clock.
        - end_time : float, optional
        end of time slice; units are those of the experiment clock.
        - calib_time: float
        experiment_clock time (in seconds) at calibration.

        Returns
        -------
        list of Records
        """
        # TODO: implement this

    @property
    def max_samples(self) -> int:
        """Maximum number of samples available at any given time."""
        return int(self.max_buflen * self.device_spec.sample_rate)

    def get_latest_data(self) -> List[Record]:
        """Pull all available samples in the buffer.

        The number of items returned depends on the size of the configured
        max_buflen as well as the time since the last data pull."""

        samples, timestamps = self.inlet.pull_chunk(
            max_samples=self.max_samples)

        return [
            Record(data=sample, timestamp=timestamps[i], rownum=None)
            for i, sample in enumerate(samples)
        ]

    def convert_time(self, experiment_clock: Clock, timestamp: float) -> float:
        """
        Convert a timestamp from the experiment clock to the acquisition clock.
        Used for querying the acquisition data for a time slice.

        Parameters:
        ----------
        - experiment_clock : clock used to generate the timestamp
        - timestamp : timestamp from the experiment clock

        Returns: corresponding timestamp for the acquistion clock
        """

        # experiment_time = pylsl.local_clock() - offset
        return timestamp + self.clock_offset(experiment_clock)

    def get_data_seconds(self, seconds: int) -> List[Record]:
        """Returns the last n second of data"""
        assert seconds <= self.max_buflen, f"Seconds can't exceed {self.max_buflen}"

        sample_count = seconds * self.device_spec.sample_rate
        records = self.get_latest_data()

        start_index = 0 if len(
            records) > sample_count else len(records) - sample_count
        return records[start_index:]

    def get_data_len(self) -> int:
        """Total amount of data recorded. This is a calculated value and may not be precise."""
        if self.first_sample:
            _, first_sample_time = self.first_sample
            _, stamp = self.inlet.pull_sample()
            return (stamp - first_sample_time) * self.device_spec.sample_rate
        return None

    @property
    def device_info(self):
        """Get the latest device_info."""
        return DeviceInfo(self.device_spec.sample_rate,
                          self.device_spec.channels, self.device_spec.name)

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

    @property
    def offset(self):
        """Offset in seconds from the start of acquisition to calibration
        trigger."""
        # first sample time is 0
        # convert_time at calib - first_sample time
        # TODO: how do we get the sample time at calib
        log.debug("Acquisition offset called")
        return 0

    def cleanup(self):
        """Perform any necessary cleanup."""


def device_from_metadata(metadata: StreamInfo) -> DeviceSpec:
    """Create a device_spec from the data stream metadata."""
    return DeviceSpec(name=metadata.name(),
                      channels=channel_names(metadata),
                      sample_rate=metadata.nominal_srate(),
                      content_type=metadata.type())


def default_stream(max_buflen: int) -> Tuple[StreamInlet, DeviceSpec]:
    """Connect to the default query stream. Used when no device_spec is
    provided.

    Parameters:
    -----------
    - max_buflen : maximum length, in seconds, for the stream to buffer.

    Returns:
    --------
    (stream_inlet, device_spec) tuple where device_spec is created from
    the stream metadata.
    """
    streams = resolve_stream('type', DEFAULT_DEVICE_TYPE)
    if not streams:
        raise Exception(
            f'LSL Stream not found for content type {DEFAULT_DEVICE_TYPE}')
    inlet = StreamInlet(streams[0], max_buflen=max_buflen)
    device_spec = device_from_metadata(inlet.info())
    return (inlet, device_spec)


def device_stream(device_spec: DeviceSpec, max_buflen: int) -> StreamInlet:
    """Connect to the LSL stream for the given device.

    Parameters:
    -----------
    - device_spec : info about the device to which to connect.
    - max_buflen : maximum length, in seconds, for the stream to buffer.

    Returns:
    --------
    stream_inlet, device_spec tuple
    """
    assert device_spec, "device_spec is required"
    streams = resolve_stream('type', device_spec.content_type)
    if not streams:
        raise Exception(
            f'LSL Stream not found for content type {device_spec.content_type}'
        )

    # TODO: if multiple streams are encountered search by name.
    inlet = StreamInlet(streams[0], max_buflen=max_buflen)
    check_device(device_spec, inlet.info())
    return inlet
