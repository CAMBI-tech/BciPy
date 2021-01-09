# pylint: disable=fixme,invalid-name
"""Used to find a protocol or device."""

from bcipy.acquisition.protocols.device import Device
from bcipy.acquisition.protocols.dsi.dsi_device import DsiDevice
from bcipy.acquisition.protocols.lsl.lsl_device import LslDevice
from bcipy.acquisition.protocols.dsi.dsi_protocol import DsiProtocol
from bcipy.acquisition.protocols.device_protocol import DeviceProtocol
from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.devices import DeviceSpec

# TODO: rename find_connector?
def find_device(device_spec: DeviceSpec,
                connection_method: ConnectionMethod) -> Device:
    """Find the first matching connector for the given device and
    connection method.

    Parameters
    ----------
        device_spec - details about the hardware device
        connection_method - method used to connect to the device
    Returns
    -------
        Device constructor
    """

    connector = next(conn for conn in Device.subclasses
                     if conn.supports(device_spec, connection_method))
    if not connector:
        raise ValueError(
            f'{connection_method} client for device {device_spec.name} is not supported'
        )
    return connector


def find_protocol(device_spec: DeviceSpec,
                  connection_method: ConnectionMethod = ConnectionMethod.TCP
                  ) -> DeviceProtocol:
    """Find the DeviceProtocol instance for the given DeviceSpec."""
    device_protocol = next(
        protocol for protocol in DeviceProtocol.subclasses
        if protocol.supports(device_spec, connection_method))
    if not device_protocol:
        raise ValueError(
            f"{device_spec.name} over {connection_method.name} is not supported."
        )
    return device_protocol(device_spec)
