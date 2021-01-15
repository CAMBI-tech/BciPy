# pylint: disable=fixme
"""Defines the driver for the Device for communicating with
LabStreamingLayer (LSL)."""
import logging

from typing import List, Dict, Any
import pylsl
from bcipy.acquisition.protocols.connector import Connector
from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.devices import DeviceSpec

log = logging.getLogger(__name__)

LSL_TIMESTAMP = 'LSL_timestamp'


class Marker():
    """Data class which wraps a LSL marker; data pulled from a marker stream is
    a tuple where the first item is a list of channels and second item is the
    timestamp. Assumes that marker inlet only has a single channel."""

    def __init__(self, data=(None, None)):
        super(Marker, self).__init__()
        self.channels, self.timestamp = data

    @classmethod
    def empty(cls):
        """Creates an empty Marker."""
        return Marker()

    def __repr__(self):
        return f"<value: {self.trg}, timestamp: {self.timestamp}>"

    @property
    def is_empty(self):
        """Test to see if the current marker is empty."""
        return self.channels is None or self.timestamp is None

    @property
    def trg(self):
        """Get the trigger."""
        # pylint: disable=unsubscriptable-object
        return self.channels[0] if self.channels else None


def inlet_name(inlet) -> str:
    """Returns the name of a pylsl streamInlet."""
    name = '_'.join(inlet.info().name().split())
    return name.replace('-', '_')


def channel_names(stream_info: pylsl.StreamInfo) -> List[str]:
    """Extracts the channel names from the LSL Stream metadata."""
    channels = []
    if stream_info.desc().child("channels").empty():
        return channels

    channel = stream_info.desc().child("channels").child("channel")
    for _ in range(stream_info.channel_count()):
        channel_name = channel.child_value("label")
        channels.append(channel_name)
        channel = channel.next_sibling()

    return channels


def rename_items(items: List[str], rules: Dict[str, str]) -> None:
    """Renames items based on the provided rules.
    Parameters
    ----------
        items - list of items ; values will be mutated
        rules - change key -> value
    """
    for key, val in rules.items():
        if key in items:
            items[items.index(key)] = val


class LslConnector(Connector):
    """Connects to any device streaming data through the LabStreamingLayer lib.

    Parameters
    ----------
        connection_params : dict
            parameters used to connect with the server.
        device_spec : DeviceSpec
            details about the device data that is being streamed over LSL.
        include_lsl_timestamp: bool, optional
            if True, appends the LSL_timestamp to each sample.
        include_marker_streams : bool, optional
            if True, listens for marker streams and merges them with the data
            stream based on its LSL timestamp. The additional columns use the
            stream names.
        rename_rules : dict, optional
            rules for renaming channels
    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments
    def __init__(self,
                 connection_params: Dict[str, Any] = {},
                 device_spec: DeviceSpec = None,
                 include_lsl_timestamp: bool = False,
                 include_marker_streams: bool = False,
                 rename_rules: Dict[str, str] = None):
        super(LslConnector, self).__init__(connection_params, device_spec)
        assert device_spec, "DeviceSpec is required"
        self._appended_channels = []

        if include_lsl_timestamp:
            self._appended_channels.append(LSL_TIMESTAMP)

        self._inlet = None
        self._marker_inlets = []
        self.include_marker_streams = include_marker_streams
        self.rename_rules = rename_rules or {}
        # There can be 1 current marker for each marker channel.
        self.current_markers = {}

    @classmethod
    def supports(cls, device_spec: DeviceSpec,
                 connection_method: ConnectionMethod) -> bool:
        # The content_type requirement can be relaxed if the `connect` method is refactored
        # to resolve the LSL stream based on the device_spec.
        return connection_method == ConnectionMethod.LSL

    @property
    def name(self):
        if 'stream_name' in self.connection_params:
            return self.connection_params['stream_name']
        if self._inlet and self._inlet.info().name():
            return self._inlet.info().name()
        return self.device_spec.name

    def connect(self):
        """Connect to the data source."""
        # Streams can be queried by name, type (xdf file format spec), and
        # other metadata.

        # NOTE: According to the documentation this is a blocking call that can
        # only be performed on the main thread in Linux systems. So far testing
        # seems fine when done in a separate multiprocessing.Process.
        eeg_streams = pylsl.resolve_stream('type',
                                           self.device_spec.content_type)
        marker_streams = pylsl.resolve_stream(
            'type', 'Markers') if self.include_marker_streams else []

        assert eeg_streams, f"One or more {self.device_spec.content_type} streams must be present"

        self._inlet = pylsl.StreamInlet(eeg_streams[0])
        self._marker_inlets = [
            pylsl.StreamInlet(inlet) for inlet in marker_streams
        ]

        # initialize the current_markers for each marker stream.
        for inlet in self._marker_inlets:
            self.current_markers[inlet_name(inlet)] = Marker.empty()

    def acquisition_init(self):
        """Initialization step. Reads the channel and data rate information
        sent by the server and sets the appropriate instance variables.
        """
        assert self._inlet is not None, "Connect call is required."
        metadata = self._inlet.info()
        self.log_info(metadata)

        channels = channel_names(metadata)
        # Confirm that provided channels match metadata, or meta is empty.
        if channels and self.device_spec.channels != channels:
            print(f"device channels: {channels}")
            print(self.device_spec.channels)
            raise Exception(f"Channels read from the device do not match "
                            "the provided parameters.")
        assert len(self.device_spec.channels) == metadata.channel_count(
        ), "Channel count error"

        if self.device_spec.sample_rate != metadata.nominal_srate():
            raise Exception("Sample frequency read from device does not match "
                            "the provided parameter")

        rename_items(self.channels, self.rename_rules)

        self.channels.extend(self._appended_channels)
        self.channels.extend(self.marker_stream_names())

        assert len(self.channels) == len(set(
            self.channels)), "Duplicate channel names are not allowed"

    def log_info(self, metadata: pylsl.StreamInfo) -> None:
        """Log information about the current connections."""
        log.debug(metadata.as_xml())
        for marker_inlet in self._marker_inlets:
            log.debug("Streaming from marker inlet: %s",
                      inlet_name(marker_inlet))

    def marker_stream_names(self) -> List[str]:
        return list(map(inlet_name, self._marker_inlets))

    def read_data(self):
        """Reads the next packet and returns the sensor data.

        Returns
        -------
            list with an item for each channel.
        """
        sample, timestamp = self._inlet.pull_sample()

        # Useful for debugging.
        if LSL_TIMESTAMP in self._appended_channels:
            sample.append(timestamp)

        for marker_inlet in self._marker_inlets:
            name = inlet_name(marker_inlet)
            marker = self.current_markers.get(name, Marker().empty())

            # Only attempt to retrieve a marker from the inlet if we have
            # merged the last one with a sample.
            if marker.is_empty:
                # A timeout of 0.0 only returns a sample if one is buffered for
                # immediate pickup. Without a timeout, this is a blocking call.
                marker_data = marker_inlet.pull_sample(timeout=0.0)
                marker = Marker(marker_data)
                self.current_markers[name] = marker

                if not marker.is_empty:
                    log.debug(
                        "Read marker %s from %s; current sample time: %s",
                        marker, name, timestamp)

            trg = "0"
            if not marker.is_empty and timestamp >= marker.timestamp:
                trg = marker.trg
                log.debug(("Appending %s marker %s to sample at time %s; ",
                           "time diff: %s"), name, marker, timestamp,
                          timestamp - marker.timestamp)
                self.current_markers[name] = Marker.empty()  # clear current

            # Add marker field to sample
            sample.append(trg)

        return sample
