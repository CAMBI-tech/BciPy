import logging

import pylsl
from acquisition.protocols.device import Device

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

TRG = "TRG"
LSL_TIMESTAMP = 'LSL_timestamp'


def empty_marker():
    return (None, None)


def inlet_name(inlet):
    return inlet.info().name()


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
    """

    def __init__(self, connection_params, fs=None, channels=None,
                 include_lsl_timestamp=False):
        super(LslDevice, self).__init__(connection_params, fs, channels)

        self._appended_channels = []
        if include_lsl_timestamp:
            self._appended_channels.append(LSL_TIMESTAMP)

        self._marker_channels = []
        self._inlet = None
        # There can be 1 current marker for each marker channel.
        self._current_markers = {}
        self._current_marker = (None, None)

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
            self._current_markers[inlet_name(inlet)] = empty_marker()

        # self._marker_inlet = pylsl.StreamInlet(marker_streams[0])

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

        for i, inlet in enumerate(self._marker_inlets):
            col = inlet_name(inlet)
            if i == len(self._marker_inlets) - 1:
                col = 'TRG'
            channels.append(col)

        return channels

    def has_current_marker(self, inlet_name):
        # return self._current_marker and self._current_marker[0] is not None
        return self._current_markers[inlet_name] and \
            self._current_markers[inlet_name][0] is not None

    def set_current_marker(self, inlet_name, value):
        self._current_markers[inlet_name] = value

    def clear_current_marker(self, inlet_name):
        # self._current_marker = (None, None)
        self.set_current_marker(inlet_name, empty_marker())

    def current_marker_trg(self, inlet_name):
        # Current marker is a tuple where first item is a list of channels
        # and second item is the timestamp.
        return self._current_markers[inlet_name][0][0]
        # return self._current_marker[0][0]

    def current_marker_ts(self, inlet_name):
        # return self._current_marker[1]
        return self._current_markers[inlet_name][1]

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

            # Only attempt to retrieve a marker from the inlet if we have
            # merged the last one with a sample.
            if not self.has_current_marker(name):
                # A timeout of 0.0 only returns a sample if one is buffered for
                # immediate pickup. Without a timeout, this is a blocking call.
                self.set_current_marker(name,
                                        marker_inlet.pull_sample(timeout=0.0))

                if self.has_current_marker(name):
                    logging.debug((f"Read marker from {name} with timestamp: "
                                   f"{self.current_marker_ts(name)}; "
                                   f"current sample time: {timestamp}"))

            trg = "0"
            if self.has_current_marker(name) and timestamp >= self.current_marker_ts(name):
                marker_ts = self.current_marker_ts(name)
                trg = self.current_marker_trg(name)
                logging.debug((f"Appending {name} marker at timestamp: "
                               f"{marker_ts} to sample at time {timestamp}; "
                               f"time diff: {timestamp - marker_ts}"))
                self.clear_current_marker(name)

            # Add marker field to sample
            sample.append(trg)

        return sample
