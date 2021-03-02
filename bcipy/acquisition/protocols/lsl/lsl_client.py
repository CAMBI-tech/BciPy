"""DataAcquisitionClient for LabStreamingLayer data sources."""
import logging
from typing import Tuple

from pylsl import StreamInfo, StreamInlet, resolve_stream

from bcipy.acquisition.device_info import DeviceInfo
from bcipy.acquisition.devices import DEFAULT_DEVICE_TYPE, DeviceSpec
from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.protocols.lsl.lsl_connector import (channel_names,
                                                           check_device)
from bcipy.acquisition.protocols.lsl.lsl_recorder import LslRecorder

log = logging.getLogger(__name__)


class LslAcquisitionClient:
    """Data Acquisition Client for devices streaming data using Lab Streaming
    Layer. Its primary use is dynamically querying streaming data in realtime.

    TODO: support multiple devices.

    Parameters:
    -----------
        max_buflen - the maximum length, in seconds, of data to be queried.
            For the RSVP paradigm, this should be calculated based on the
            total inquiry length.
        device_spec - spec for the device from which to query data; if
            missing, this class will attempt to find the first EEG stream.
        append_timestamps - if True appends the LSL timestamp as an additional
            column in any queried data.
        save_directory - if present, recordings of the data will be persisted
            here.
    """

    def __init__(self,
                 max_buflen: int,
                 device_spec: DeviceSpec = None,
                 append_timestamps: bool = True,
                 save_directory: str = None):
        super().__init__()
        if device_spec:
            assert ConnectionMethod.LSL in device_spec.connection_methods, "Only LSL devices allowed."
            assert device_spec.sample_rate > 0, "Marker streams may be recorded but not queried."
        self.device_spec = device_spec
        self.max_buflen = max_buflen
        self.append_timestamps = append_timestamps

        self.inlet = None
        self.first_sample = None
        self.recorder = LslRecorder(path=save_directory) if save_directory else None

    def start_acquisition(self) -> None:
        """Connect to the datasource and start acquiring data."""
        if self.recorder:
            self.recorder.start()

        # TODO: Should we resolve all streams and query by name?
        if self._connect_to_query_stream():
            log.info(self.inlet.info().as_xml())
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

    def get_data(self, start=None, end=None, field='_rowid_'):
        """Queries the buffer by field.

        Parameters
        ----------
            start : number, optional
                start of time slice; units are those of the acquisition clock.
            end : float, optional
                end of time slice; units are those of the acquisition clock.
            field: str, optional
                field on which to query; default value is the row id.
        Returns
        -------
            list of Records
        """
        pass

    @property
    def max_samples(self) -> int:
        """Maximum number of samples available at any given time."""
        return int(self.max_buflen * self.device_spec.sample_rate)

    def get_latest_data(self):
        """Pull all available samples in the buffer.

        The number of items returned depends on the size of the configured
        max_buflen as well as the time since the last data pull. If the
        append_timestamps property is set the LSL timestamps will be appended
        as an additional column for each row."""

        samples, timestamps = self.inlet.pull_chunk(
            max_samples=self.max_samples)
        if self.append_timestamps:
            for i, sample in enumerate(samples):
                sample.append(timestamps[i])
        return samples

    def get_data_seconds(self, seconds: int):
        """Returns the last n second of data"""
        assert seconds <= self.max_buflen, f"Seconds can't exceed {self.max_buflen}"

        sample_count = seconds * self.device_spec.sample_rate
        samples = self.get_latest_data()

        if len(samples) < sample_count:
            return samples

        starting_index = len(samples) - sample_count
        return samples[starting_index:]

    def get_data_len(self):
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
            bool_val: boolean
                if True, uses a 0 offset; if False forces the calculation.
        """
        pass

    @property
    def offset(self):
        """Offset in seconds from the start of acquisition to calibration
        trigger."""
        pass

    def cleanup(self):
        """Perform any necessary cleanup."""
        pass


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
        max_buflen - maximum length, in seconds, for the stream to buffer.

    Returns:
    --------
        stream_inlet, device_spec tuple where device_spec is created from
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
        device_spec - info about the device to which to connect.
        max_buflen - maximum length, in seconds, for the stream to buffer.

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