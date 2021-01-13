"""Protocol with information for generating and serving mock DSI data."""
import timeit

from typing import Any, List
from bcipy.acquisition.protocols.dsi import dsi
from bcipy.acquisition.protocols.device_protocol import DeviceProtocol
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.connection_method import ConnectionMethod


class DsiProtocol(DeviceProtocol):
    """Protocol for mocking DSI data over TCP."""

    def __init__(self, device_spec: DeviceSpec):
        super().__init__(device_spec)
        self.start_time = timeit.default_timer()

    @classmethod
    def connection_method(cls):
        return ConnectionMethod.TCP

    @classmethod
    def supports(cls, device_spec: DeviceSpec,
                 connection_method: ConnectionMethod) -> bool:
        return device_spec.name.startswith(
            'DSI') and connection_method == cls.connection_method()

    def init_messages(self) -> List[Any]:
        """Messages sent at the start of the initialization protocol when
        connecting with a TCP client. Sent before any data is acquired.
        """
        s_msg = ','.join(self.device_spec.channels).encode('ascii', 'ignore')
        sensor_msg = event_packet('SENSOR_MAP', s_msg)

        # Comma-delimited list; the frequency is the second param, which is the
        # only one we use; not sure what else is in this list.
        f_msg = (',' + str(int(self.device_spec.sample_rate))).encode(
            'ascii', 'ignore')
        freq_msg = event_packet('DATA_RATE', f_msg)

        return [sensor_msg, freq_msg]

    def encode(self, sensor_data: List[float]) -> Any:
        """Builds a binary data packet from the provided sensor data.

        Parameters
        ----------
            sensor_data: list of sensor values; len must
                match the channel_count

        Returns
        -------
        Binary data for a single packet.
        """
        # (timestamp/4 bytes, counter/1 byte, status/6 bytes) + 4 bytes for
        # every sensor/float.
        payload_length = (11 + (len(sensor_data) * 4))

        params = dict(
            type='EEG_DATA',
            payload_length=payload_length,
            number=13,  # arbitrary number
            timestamp=(timeit.default_timer() - self.start_time),
            data_counter=0,  # unused
            ADC_status=b"123456",  # unused
            sensor_data=sensor_data)

        return dsi.packet.build(params)


def event_packet(code, msg):
    """Construct an event packet with the given event code and message."""

    # calculate payload length
    event_code_bytes, sending_node_bytes, msg_len_bytes, msg_bytes = (4, 4, 4,
                                                                      len(msg))

    payload_len = sum(
        [event_code_bytes, sending_node_bytes, msg_len_bytes, msg_bytes])

    params = dict(type='EVENT',
                  payload_length=payload_len,
                  number=13,
                  event_code=code,
                  sending_node=33,
                  message_length=len(msg),
                  message=msg)

    return dsi.packet.build(params)
