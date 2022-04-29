"""Defines the driver for the Device for communicating with the DSI headset."""
import logging
import socket

from bcipy.acquisition.protocols.dsi import dsi
from bcipy.acquisition.protocols import util
from bcipy.acquisition.protocols.connector import Connector
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.connection_method import ConnectionMethod

log = logging.getLogger(__name__)


class DsiConnector(Connector):
    """Connects to a DSI device over TCP.

    Parameters
    ----------
        connection_params : dict
            parameters used to connect with the server. keys: [host, port]
        device_spec: DeviceSpec
            spec with information about the device to which to connect.
    """

    def __init__(self, connection_params, device_spec: DeviceSpec):
        """Init DsiDevice."""
        super(DsiConnector, self).__init__(connection_params, device_spec)
        assert 'host' in connection_params, "Please specify host to Device!"
        assert 'port' in connection_params, "Please specify port to Device!"
        self._channels_provided = len(self.channels) > 0
        self._socket = None

    @classmethod
    def supports(cls, device_spec: DeviceSpec,
                 connection_method: ConnectionMethod) -> bool:
        return device_spec.name.startswith(
            'DSI') and connection_method == ConnectionMethod.TCP

    @classmethod
    def connection_method(cls) -> ConnectionMethod:
        return ConnectionMethod.TCP

    @property
    def name(self):
        """DSI Name."""
        return 'DSI'

    def connect(self):
        """Connect to the data source."""
        params = self.connection_params
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
            log.debug(response.type)

            # Here we get information from the device about version etc.
            #  If interested, print the response type and message!
            if response.type == 'EVENT':
                pass
            response = self._read_packet()

        channels = response.message.split(',')
        log.debug("Channels: %s", ','.join(channels))
        if len(channels) != len(self.channels):
            raise Exception("Channels read from DSI device do not match "
                            "the provided parameters")
        response = self._read_packet()

        if response.type != 'EVENT' or response.event_code != 'DATA_RATE':
            raise Exception("Unexpected packet; expected DATA RATE Event")

        sample_freq = int(response.message.split(',')[1])
        log.debug("Sample frequency: %s", str(sample_freq))

        if sample_freq != self.fs:
            raise Exception("Sample frequency read from DSI device does not "
                            "match the provided parameter")

        # Read once more for data start
        response = self._read_packet()

    def read_data(self):
        """Read Data.

        Reads the next packet from DSI device and returns the sensor data.

        Returns
        -------
            list with an item for each channel.
        """
        sensor_data = False

        while not sensor_data:

            response = self._read_packet()

            if hasattr(response, 'sensor_data'):
                return list(response.sensor_data)
