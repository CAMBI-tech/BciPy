"""Functionality for managing multiple devices."""
import logging
from typing import Any, Dict, List, Optional

from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.exceptions import UnsupportedContentType
from bcipy.acquisition.protocols.lsl.lsl_client import LslAcquisitionClient
from bcipy.acquisition.record import Record
from bcipy.helpers.system_utils import AutoNumberEnum

log = logging.getLogger(__name__)


class ContentType(AutoNumberEnum):
    """Enum of supported acquisition device (LSL) content types. Allows for
    case-insensitive matching, as well as synonyms for some types.

    >>> ContentType(1) == ContentType.EEG
    True
    >>> ContentType('Eeg') == ContentType.EEG
    True
    """

    def __init__(self, synonyms: List[str]):
        self.synonyms = synonyms

    EEG = []
    EYETRACKER = ['gaze', 'eye_tracker']
    MARKERS = ['switch']

    @classmethod
    def _missing_(cls, value: Any):
        """Lookup function used when a value is not found."""
        value = str(value).lower()
        for member in cls:
            if member.name.lower() == value or value in member.synonyms:
                return member
        raise UnsupportedContentType(f"ContentType not supported: {value}")


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

    def __init__(self, default_content_type: ContentType = ContentType.EEG):
        self._clients: Dict[ContentType, LslAcquisitionClient] = {}
        self.default_content_type = default_content_type

    @property
    def clients(self) -> List[LslAcquisitionClient]:
        """Returns a list of the managed clients."""
        return self._clients.values()

    @property
    def clients_by_type(self) -> Dict[ContentType, LslAcquisitionClient]:
        """Returns a dict of clients keyed by their content type"""
        return self._clients

    @property
    def device_specs(self) -> List[DeviceSpec]:
        """Returns a list of DeviceSpecs for all the clients."""
        return [client.device_spec for client in self.clients]

    @property
    def device_content_types(self) -> List[ContentType]:
        """Returns a list of ContentTypes provided by the configured devices.
        """
        return self._clients.keys()

    @property
    def default_client(self) -> Optional[LslAcquisitionClient]:
        """Returns the default client."""
        return self.get_client(self.default_content_type)

    def add_client(self, client: LslAcquisitionClient):
        """Add the given client to the manager."""
        content_type = ContentType(client.device_spec.content_type)
        self._clients[content_type] = client

    def get_client(
            self, content_type: ContentType) -> Optional[LslAcquisitionClient]:
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

    def get_data_by_device(
        self,
        start: float = None,
        seconds: float = None,
        content_types: List[ContentType] = None
    ) -> Dict[ContentType, List[Record]]:
        """Get data for one or more devices. The number of samples for each
        device depends on the sample rate and may be different for item.

        Parameters
        ----------
            start - start time (acquisition clock) of data window
            seconds - duration of data to return for each device
            content_types - specifies which devices to include
        """
        output = {}
        for content_type in content_types:
            client = self.get_client(content_type)
            if client.device_spec.sample_rate > 0:
                count = round(seconds * client.device_spec.sample_rate)
                log.debug(
                    f'Need {count} records for processing {content_type.name} data'
                )
                output[content_type] = client.get_data(start=start,
                                                       limit=count)
            else:
                # Markers have an IRREGULAR_RATE.
                output[content_type] = client.get_data(start=start,
                                                       end=start + seconds)
        return output

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
