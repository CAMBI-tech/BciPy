from __future__ import absolute_import, division, print_function

import logging

import pylsl
from acquisition.protocols.device import Device

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
        fs: float, optional
            sample frequency in (Hz)
    """

    def __init__(self, connection_params, fs=None, channels=None):
        super(LslDevice, self).__init__(connection_params, fs, channels)

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
        self._marker_inlet = pylsl.StreamInlet(marker_streams[0])

    def acquisition_init(self):
        """Initialization step. Reads the channel and data rate information
        sent by the server and sets the appropriate instance variables.
        """
        assert self._inlet is not None, "Connect call is required."
        metadata = self._inlet.info()
        logging.debug(metadata.as_xml())
        logging.debug(self._marker_inlet.info().as_xml())

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
        assert len(self.channels) == (metadata.channel_count() + 1),\
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

        if not 'TRG' in channels:
            channels.append('TRG')

        return channels

    def read_data(self):
        """Reads the next packet and returns the sensor data.

        Returns
        -------
            list with an item for each channel.
        """
        sample, timestamp = self._inlet.pull_sample()
        # A timeout of 0.0 only returns a sample if one is buffered for
        # immediate pickup. Without a timeout, this is a blocking call.
        marker_sample, ts = self._marker_inlet.pull_sample(timeout=0.0)

        # Add marker field to sample.
        # TODO: consider checking the len of the sample; not sure if any of the
        #   LSL drivers include the TRG field in the sample.
        # TODO: should we compare the timestamps?
        if marker_sample is not None and int(marker_sample[0]) > 0:
            logging.debug("Marker: {}; time: {}; sample time: {}".format(
                marker_sample[0], ts, timestamp))
            sample.append(marker_sample[0])
        else:
            sample.append("0")

        return sample
