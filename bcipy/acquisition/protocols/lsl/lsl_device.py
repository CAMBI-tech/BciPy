import logging

import pylsl
from bcipy.acquisition.protocols.device import Device

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

TRG = "TRG"
LSL_TIMESTAMP = 'LSL_timestamp'


class Marker(object):
    """Data class which wraps a LSL marker; data pulled from a marker stream is
    a tuple where the first item is a list of channels and second item is the
    timestamp. Assumes that marker inlet only has a single channel."""

    def __init__(self, data=(None, None)):
        super(Marker, self).__init__()
        self.channels, self.timestamp = data

    @classmethod
    def empty(cls):
        return Marker()

    def __repr__(self):
        return f"<value: {self.trg}, timestamp: {self.timestamp}>"

    @property
    def is_empty(self):
        return self.channels is None or self.timestamp is None

    @property
    def trg(self):
        return self.channels[0] if self.channels else None


def inlet_name(inlet):
    name = '_'.join(inlet.info().name().split())
    return name.replace('-', '_')


class LslDevice(Device):
    """Driver for any device streaming data through the LabStreamingLayer lib.

    Parameters
    ----------
        connection_params : dict
            parameters used to connect with the server.
        channels: list, optional
            list of channel names
        fs: float, optional
            sample frequency in (Hz)
        include_lsl_timestamp: bool, optional
            if True, appends the LSL_timestamp to each sample.
        trg_channel_name: str, optional
            if present, the marker channel with this name is used as the TRG
            channel in the output and queried for offset; otherwise the last
            marker_channel added is used.
    """

    def __init__(self, connection_params, fs=None, channels=None,
                 include_lsl_timestamp=False,
                 trg_channel_name='BCI_Stimulus_Markers'):
        super(LslDevice, self).__init__(connection_params, fs, channels)

        self._appended_channels = []
        self.trg_channel_name = trg_channel_name

        if include_lsl_timestamp:
            self._appended_channels.append(LSL_TIMESTAMP)

        self._marker_channels = []
        self._inlet = None
        # There can be 1 current marker for each marker channel.
        self.current_markers = {}

    @property
    def name(self):
        if 'stream_name' in self._connection_params:
            return self._connection_params['stream_name']
        elif self._inlet and self._inlet.info().name():
            return self._inlet.info().name()
        return 'LSL'

    def connect(self):
        """Connect to the data source."""
        # Streams can be queried by name, type (xdf file format spec), and
        # other metadata.
        # TODO: consider using other connection_params here.

        # NOTE: According to the documentation this is a blocking call that can
        # only be performed on the main thread in Linux systems. So far testing
        # seems fine when done in a separate multiprocessing.Process.
        streams = pylsl.resolve_stream('type', 'EEG')
        marker_streams = pylsl.resolve_stream('type', 'Markers')

        assert len(streams) > 0
        assert len(marker_streams) > 0
        self._inlet = pylsl.StreamInlet(streams[0])

        self._marker_inlets = [pylsl.StreamInlet(inlet)
                               for inlet in marker_streams]

        # initialize the current_markers for each marker stream.
        for inlet in self._marker_inlets:
            self.current_markers[inlet_name(inlet)] = Marker.empty()

    def acquisition_init(self):
        """Initialization step. Reads the channel and data rate information
        sent by the server and sets the appropriate instance variables.
        """
        assert self._inlet is not None, "Connect call is required."
        metadata = self._inlet.info()
        logging.debug(metadata.as_xml())
        for mi in self._marker_inlets:
            logging.debug(f"Streaming from marker inlet: {inlet_name(mi)}")

        info_channels = self._read_channels(metadata)
        info_fs = metadata.nominal_srate()

        # If channels are not initially provided, set them from the metadata.
        # Otherwise, confirm that provided channels match metadata, or meta is
        # empty.
        if not self.channels:
            self.channels = info_channels
            assert len(self.channels) > 0, "Channels must be provided"
        else:
            if len(info_channels) > 0 and self.channels != info_channels:
                raise Exception("Channels read from the device do not match "
                                "the provided parameters")
        assert len(self.channels) == (metadata.channel_count() +
                                      len(self._appended_channels) +
                                      len(self._marker_inlets)),\
            "Channel count error"

        if not self.fs:
            self.fs = info_fs
        elif self.fs != info_fs:
            raise Exception("Sample frequency read from device does not match "
                            "the provided parameter")

    def _trigger_inlet_index(self):
        """Index of the marker inlet that should be used for the TRG column.
        If the trg_channel_name parameter is provided, and there is a marker
        inlet with this name, that is used, otherwise the last marker inlet
        is used. Returns -1 if there are no marker inlets."""

        inlet_names = [inlet_name(inlet) for inlet in self._marker_inlets]

        if self.trg_channel_name in inlet_names:
            return inlet_names.index(self.trg_channel_name)
        else:
            return len(inlet_names) - 1

    def _read_channels(self, info):
        """Read channels from the stream metadata if provided and return them
        as a list. If channels were not specified, returns an empty list.

        Parameters
        ----------
            info : pylsl.XMLElement
        Returns
        -------
            list of str
        """
        channels = []
        if info.desc().child("channels").empty():
            return channels

        ch = info.desc().child("channels").child("channel")
        for k in range(info.channel_count()):
            channels.append(ch.child_value("label"))
            ch = ch.next_sibling()

        for ac in self._appended_channels:
            channels.append(ac)

        trg_marker_index = self._trigger_inlet_index()
        for i, inlet in enumerate(self._marker_inlets):
            col = inlet_name(inlet)
            if i == trg_marker_index:
                col = 'TRG'
            channels.append(col)

        return channels

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
                    logging.debug((f"Read marker {marker} from {name}; "
                                   f"current sample time: {timestamp}"))

            trg = "0"
            if not marker.is_empty and timestamp >= marker.timestamp:
                trg = marker.trg
                logging.debug((f"Appending {name} marker {marker} "
                               f"to sample at time {timestamp}; "
                               f"time diff: {timestamp - marker.timestamp}"))
                self.current_markers[name] = Marker.empty()  # clear current

            # Add marker field to sample
            sample.append(trg)

        return sample
