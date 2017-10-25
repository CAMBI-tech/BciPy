from __future__ import absolute_import, division, print_function

import logging

import pylsl
from daq.protocols.device import Device

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class LslDevice(Device):
    """Driver for any device streaming data through the LabStreamingLayer lib.

    Parameters
    ----------
        connection_params : dict
            parameters used to connect with the server.
        channels: list, optional
            list of channel names
        hz: float, optional
            sample frequency in (Hz)
    """

    def __init__(self, connection_params, hz=None, channels=None):
        super(LslDevice, self).__init__(connection_params, hz, channels)

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

        # NOTE: this is a blocking call that can only be performed on the
        # main thread in Linux systems.
        streams = pylsl.resolve_stream('type', 'EEG')

        assert len(streams) > 0
        self._inlet = pylsl.StreamInlet(streams[0])

    def acquisition_init(self):
        """Initialization step. Reads the channel and data rate information
        sent by the server and sets the appropriate instance variables.
        """
        assert self._inlet is not None, "Connect call is required."
        metadata = self._inlet.info()
        logging.debug(metadata.as_xml())

        info_channels = self._read_channels(metadata)
        info_hz = metadata.nominal_srate()

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
        assert len(self.channels) == metadata.channel_count(), "Channel count error"  # noqa

        if not self.hz:
            self.hz = info_hz
        elif self.hz != info_hz:
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
        return channels

    def read_data(self):
        """Reads the next packet and returns the sensor data.

        Returns
        -------
            list with an item for each channel.
        """
        sample, timestamp = self._inlet.pull_sample()
        return sample
