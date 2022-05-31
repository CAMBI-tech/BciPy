"""Interface for creating new device drivers."""
from bcipy.acquisition.device_info import DeviceInfo
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.connection_method import ConnectionMethod


class Connector:
    """Base class for device-specific behavior.

    Parameters
    ----------
        connection_params : dict
            Parameters needed to connect with the given device
        device_spec: DeviceSpec
            spec with information about the device to which to connect.
    """
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        """Any subclasses will be automatically registered."""
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

    # pylint: disable=invalid-name
    def __init__(self, connection_params: dict, device_spec: DeviceSpec):
        self.connection_params = connection_params
        self.device_spec = device_spec
        self.fs = device_spec.sample_rate
        self.channels = device_spec.channels.copy()

    @classmethod
    def supports(cls, device_spec: DeviceSpec,
                 connection_method: ConnectionMethod) -> bool:
        return False

    @classmethod
    def connection_method(cls) -> ConnectionMethod:
        return None

    @property
    def name(self):
        """Device name to be written to the output."""
        raise NotImplementedError('Subclass must define a name property')

    @property
    def device_info(self) -> DeviceInfo:
        """Information about the acquisition parameters. Should be called after
        acquisition_init for those devices which set this information. Note that
        DeviceInfo may differ from the DeviceSpec if additional information is
        added by the Connector."""
        device_name = self.name if not callable(self.name) else self.name()
        return DeviceInfo(fs=self.fs, channels=self.channels, name=device_name)

    def connect(self):
        """Connect to the data source."""
        pass

    def acquisition_init(self):
        """Initialization step. Depending on the protocol, this may involve
        reading header information and setting the appropriate instance
        properties or writing to the server to set params (ex. sampling freq).
        """
        pass

    def read_data(self):
        """Read the next sensor data record from the data source.

        Returns
        -------
            list with a float for each channel.
        """
        raise NotImplementedError(
            'Subclass must define the read_sensor_data method')

    def disconnect(self):
        """Optional method to disconnect from the device and do any cleanup"""
        pass
