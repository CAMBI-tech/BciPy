"""Functionality for managing multiple devices."""
import logging
from typing import Dict, List, Optional
from bcipy.acquisition.protocols.lsl.lsl_client import LslAcquisitionClient
from bcipy.acquisition.devices import DeviceSpec

log = logging.getLogger(__name__)


class ClientManager():
    """Manages multiple acquisition clients. This class can also act as an
    acquisition client. If used in this way, it dispatches to the managed
    client with the default_client_type.

    >>> from bcipy.acquisition import LslAcquisitionClient
    >>> from bcipy.acquisition.devices import DeviceSpec
    >>> spec = DeviceSpec('Testing', ['ch1', 'ch2', 'ch3'], 60.0, 'EEG')
    >>> manager = ClientManager()
    >>> eeg_client = LslAcquisitionClient(device_spec=spec)
    >>> manager.add_client(eeg_client)
    >>> manager.device_spec == spec
    True

    Parameters
    ----------
        default_content_type - used for dispatching calls to an LslClient.
    """

    def __init__(self, default_content_type: str = 'EEG'):
        self._clients: Dict[str, LslAcquisitionClient] = {}
        self.default_content_type = default_content_type

    @property
    def clients(self) -> List[LslAcquisitionClient]:
        """Returns a list of the managed clients."""
        return self._clients.values()

    @property
    def device_specs(self) -> List[DeviceSpec]:
        """Returns a list of DeviceSpecs for all the clients."""
        return [client.device_spec for client in self.clients]

    @property
    def default_client(self) -> Optional[LslAcquisitionClient]:
        """Returns the default client."""
        return self.get_client(self.default_content_type)

    def add_client(self, client: LslAcquisitionClient):
        """Add the given client to the manager."""
        self._clients[client.device_spec.content_type] = client

    def get_client(self, content_type: str) -> Optional[LslAcquisitionClient]:
        """Get client by content type"""
        return self._clients.get(content_type, None)

    def start_acquisition(self):
        """Start acquiring data for all clients"""
        for client in self.clients:
            log.info(f"Connecting to {client.device_spec.name}...")
            client.start_acquisition()

    def stop_acquisition(self):
        """Stop acquiring data for all clients"""
        for client in self.clients:
            client.stop_acquisition()

    def cleanup(self):
        """Perform any cleanup tasks"""
        for client in self.clients:
            client.cleanup()

    def __getattr__(self, name):
        """Dispatch unknown properties and methods to the client with the
        default content type."""
        client = self.default_client
        if client:
            return client.__getattribute__(name)
        raise AttributeError(f"Missing attribute: {name}")
