import logging
import socket

import acquisition.protocols.dsi.dsi as dsi
import acquisition.protocols.util as util
from acquisition.protocols.device import Device

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s', )


class DsiDevice(Device):
    """Driver for the DSI device.

    Parameters
    ----------
        connection_params : dict
            parameters used to connect with the server. keys: [host, port]
        channels: list, optional
            list of channel names
        fs: float, optional
            sample frequency in (Hz)
    """

    def __init__(self, connection_params, fs=dsi.DEFAULT_FS, channels=dsi.DEFAULT_CHANNELS):
        """Init DsiDevice."""
        super(DsiDevice, self).__init__(connection_params, fs, channels)
        assert 'host' in connection_params, "Please specify host to Device!"
        assert 'port' in connection_params, "Please specify port to Device!"
        self._channels_provided = len(channels) > 0
        self._socket = None
        self.channels = channels

    @property
    def name(self):
        """DSI Name."""
        return 'DSI'

    def connect(self):
        """Connect to the data source."""
        params = self._connection_params
        address = (params['host'], params['port'])
        self._socket = socket.create_connection(address)
        self._socket.settimeout(None)

    def disconnect(self):
        self._socket.close()

    def _read_packet(self):
        """Read a single packet from the data source.

        Returns
        -------
            dict-like object
        """

        assert self._socket is not None, \
            "Socket isn't started, cannot read DSI packet!"

        # Reads the header to get the payload length, then reads the payload.
        header_buf = util.receive(self._socket, dsi.HEADER_LEN)
        header = dsi.header.parse(header_buf)
        payload_buf = util.receive(self._socket, header.payload_length)
        return dsi.packet.parse(header_buf + payload_buf)

    def acquisition_init(self):
        """Initialization step.

        Reads the channel and data rate information
        sent by the server and sets the appropriate instance variables.
        """

        response = self._read_packet()
        # Read packets until we encounter the headers we care about.
        while response.type != 'EVENT' or response.event_code != 'SENSOR_MAP':
            if response.type == 'EEG_DATA':
                raise Exception('EEG Data was encountered; expected '
                                'initialization headers.')
            logging.debug(response.type)

            # Here we get information from the device about version etc.
            #  If interested, print the response type and message!
            if response.type == 'EVENT':
                pass
            response = self._read_packet()

        channels = response.message.split(',')
        logging.debug("Channels: " + ','.join(channels))
        if self._channels_provided and len(channels) != len(self.channels):
            raise Exception("Channels read from DSI device do not match "
                            "the provided parameters")
        else:
            self.channels = dsi.DEFAULT_CHANNELS
        response = self._read_packet()

        if response.type != 'EVENT' or response.event_code != 'DATA_RATE':
            raise Exception("Unexpected packet; expected DATA RATE Event")

        fs = int(response.message.split(',')[1])
        logging.debug("Sample frequency: " + str(fs))

        if fs != self.fs:
            raise Exception("Sample frequency read from DSI device does not "
                            "match the provided parameter")

        # Read once more for data start
        response = self._read_packet()

    def read_data(self):
        """Read Data.

        Reads he next packet from DSI device and returns the sensor data.

        Returns
        -------
            list with an item for each channel.
        """
        sensor_data = False

        while not sensor_data:

            response = self._read_packet()

            if hasattr(response, 'sensor_data'):
                return list(response.sensor_data)
