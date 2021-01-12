# pylint: disable=fixme,invalid-name
"""Used to find a protocol or device."""

from typing import Dict, Callable
from bcipy.acquisition.protocols.connector import Connector
from bcipy.acquisition.protocols.dsi.dsi_connector import DsiConnector
from bcipy.acquisition.protocols.lsl.lsl_connector import LslConnector
from bcipy.acquisition.protocols.dsi.dsi_protocol import DsiProtocol
from bcipy.acquisition.protocols.device_protocol import DeviceProtocol
from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.devices import DeviceSpec


def find_connector(device_spec: DeviceSpec, connection_method: ConnectionMethod
                   ) -> Callable[[DeviceSpec, Dict], Connector]:
    """Find the first matching connector for the given device and
    connection method.

    Parameters
    ----------
        device_spec - details about the hardware device
        connection_method - method used to connect to the device
    Returns
    -------
        Connector constructor
    """

    connector = next(conn for conn in Connector.subclasses
                     if conn.supports(device_spec, connection_method))
    if not connector:
        raise ValueError(
            f'{connection_method} client for device {device_spec.name} is not supported'
        )
    return connector


def make_connector(device_spec: DeviceSpec,
                   connection_method: ConnectionMethod,
                   connection_params: dict) -> Connector:
    """Find and construct a Connector for the given device and connection method.

    Parameters
    ----------
        device_spec - details about the hardware device.
        connection_method - method used to connect to the device.
        connection_params - parameters specific to the relevant Connector, such
          as host and port information (for a TCP connector).
    Returns
    -------
        Connector instanct
    """
    connector = find_connector(device_spec, connection_method)
    return connector(connection_params=connection_params,
                     device_spec=device_spec)


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
