"""Defines the Protocol data structure"""
from typing import Any, List
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.connection_method import ConnectionMethod


class DeviceProtocol:
    """DeviceProtocols are used in Data Servers to mock the communication
    patterns expected for the given device. For instance, in a TCP server,
    there is often a handshake prior to data acquisition where the device will
    output relevant header information. This class also has the ability to
    encode sample sensor data to a binary format."""
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        """Any subclasses will be automatically registered."""
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

    def __init__(self, device_spec: DeviceSpec):
        super().__init__()
        if not self.__class__.supports(device_spec,
                                       self.__class__.connection_method()):
            raise ValueError(f"Device not supported: {device_spec.name}")
        self.device_spec = device_spec

    @classmethod
    def supports(cls, device_spec: DeviceSpec,
                 connection_method: ConnectionMethod) -> bool:
        return False

    @classmethod
    def connection_method(cls):
        return None

    @property
    def sample_rate(self) -> float:
        """Sample frequency in Hz."""
        return self.device_spec.sample_rate

    @property
    def channels(self) -> List[str]:
        """List of channel names."""
        return self.device_spec.channels

    def init_messages(self) -> List[Any]:
        """Messages sent at the start of the initialization protocol when
        connecting with a TCP client. Sent before any data is acquired.
        """
        return []

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
        return sensor_data
