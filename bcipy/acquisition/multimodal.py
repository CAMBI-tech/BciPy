# mypy: disable-error-code="arg-type"
"""Functionality for managing multiple devices."""
import logging
from typing import Any, Dict, List, Optional

from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.exceptions import (InsufficientDataException,
                                          UnsupportedContentType)
from bcipy.acquisition.protocols.lsl.lsl_client import LslAcquisitionClient
from bcipy.acquisition.record import Record
from bcipy.config import SESSION_LOG_FILENAME
from bcipy.helpers.utils import AutoNumberEnum

logger = logging.getLogger(SESSION_LOG_FILENAME)


class ContentType(AutoNumberEnum):
    """Enum of supported acquisition device (LSL) content types.

    Allows for case-insensitive matching, as well as synonyms for some types.

    Examples:
        >>> ContentType(1) == ContentType.EEG
        True
        >>> ContentType('Eeg') == ContentType.EEG
        True
    """

    def __init__(self, synonyms: List[str]):
        self.synonyms = synonyms

    EEG: List[str] = []
    EYETRACKER: List[str] = ['gaze', 'eye_tracker']
    MARKERS: List[str] = ['switch']

    @classmethod
    def _missing_(cls, value: Any) -> 'ContentType':
        """Lookup function used when a value is not found.

        This method enables case-insensitive matching and allows for synonyms.

        Args:
            value (Any): The value to lookup, which will be converted to a string and lowercased.

        Returns:
            ContentType: The matching ContentType enum member.

        Raises:
            UnsupportedContentType: If no matching content type is found.
        """
        value = str(value).lower()
        for member in cls:  # type: ignore
            if member.name.lower() == value or value in member.synonyms:  # type: ignore
                return member
        raise UnsupportedContentType(f"ContentType not supported: {value}")


class ClientManager():
    """Manages multiple acquisition clients.

    This class can also act as an acquisition client. If used in this way,
    it dispatches to the managed client with the `default_client_type`.

    Args:
        default_content_type (ContentType, optional): The default content type
            to use for dispatching calls to an `LslClient`. Defaults to `ContentType.EEG`.
    """

    inlet: Any = None  # To satisfy mypy, as ClientManager can act as an LslAcquisitionClient

    def __init__(self, default_content_type: ContentType = ContentType.EEG) -> None:  # type: ignore
        self._clients: Dict[ContentType, LslAcquisitionClient] = {}
        self.default_content_type = default_content_type

    @property
    def clients(self) -> List[LslAcquisitionClient]:
        """Returns a list of the managed clients."""
        return list(self._clients.values())

    @property
    def clients_by_type(self) -> Dict[ContentType, LslAcquisitionClient]:
        """Returns a dictionary of clients keyed by their content type."""
        return self._clients

    @property
    def device_specs(self) -> List[DeviceSpec]:
        """Returns a list of `DeviceSpec` objects for all the clients."""
        return [client.device_spec for client in self.clients if client.device_spec]  # type: ignore

    @property
    def device_content_types(self) -> List[ContentType]:
        """Returns a list of `ContentType` enums provided by the configured devices.
        """
        return list(self._clients.keys())

    @property
    def active_device_content_types(self) -> List[ContentType]:
        """Returns a list of `ContentType` enums provided by the active configured
        devices."""
        return [
            content_type for content_type, client in self._clients.items()
            if client.device_spec and client.device_spec.is_active  # type: ignore
        ]

    @property
    def default_client(self) -> Optional[LslAcquisitionClient]:
        """Returns the default client."""
        return self.get_client(self.default_content_type)

    def add_client(self, client: LslAcquisitionClient) -> None:  # type: ignore
        """Adds the given client to the manager.

        Args:
            client (LslAcquisitionClient): The client instance to add.
        """
        content_type = ContentType(client.device_spec.content_type)  # type: ignore
        self._clients[content_type] = client

    def get_client(
            self, content_type: ContentType) -> Optional[LslAcquisitionClient]:
        """Retrieves a client by its content type.

        Args:
            content_type (ContentType): The content type of the client to retrieve.

        Returns:
            Optional[LslAcquisitionClient]: The `LslAcquisitionClient` instance if found,
                                            otherwise None.
        """
        return self._clients.get(content_type, None)

    def start_acquisition(self) -> None:  # type: ignore
        """Starts data acquisition for all clients."""
        for client in self.clients:
            logger.info(f"Connecting to {client.device_spec.name}...")  # type: ignore
            client.start_acquisition()

    def stop_acquisition(self) -> None:  # type: ignore
        """Stops data acquisition for all clients."""
        logger.info("Stopping acquisition...")
        for client in self.clients:
            client.stop_acquisition()

    def get_data_by_device(
        self,
        start: Optional[float] = None,
        seconds: Optional[float] = None,
        content_types: Optional[List[ContentType]] = None,
        strict: bool = True
    ) -> Dict[ContentType, List[Record]]:
        """Retrieves data for one or more devices within a specified time window.

        The number of samples for each device depends on the sample rate and may
        differ. The actual start time will be adjusted by the `static_offset`
        configured for each device.

        Args:
            start (Optional[float]): Start time (acquisition clock) of the data window.
            seconds (Optional[float]): Duration of data to return for each device.
            content_types (Optional[List[ContentType]], optional): Specifies which
                devices to include. If None, data for all types is returned.
                Defaults to None.
            strict (bool, optional): If True, raises an `InsufficientDataException`
                if the number of returned records is less than the requested number.
                Defaults to True.

        Returns:
            Dict[ContentType, List[Record]]: A dictionary where keys are `ContentType`
                                           and values are lists of `Record` objects.

        Raises:
            InsufficientDataException: If `strict` is True and the returned data count
                                       is less than the requested count for a device.
        """
        output: Dict[ContentType, List[Record]] = {}
        if not content_types:
            content_types = self.device_content_types
        for content_type in content_types:
            name = content_type.name
            client = self.get_client(content_type)

            if client and client.device_spec:
                adjusted_start = start + client.device_spec.static_offset
                if client.device_spec.sample_rate > 0:
                    count = round(seconds * client.device_spec.sample_rate)
                    logger.info(f'Need {count} records for processing {name} data')
                    output[content_type] = client.get_data(start=adjusted_start,
                                                           limit=count)
                    data_count = len(output[content_type])
                    if strict and data_count < count:
                        msg = f'Needed {count} {name} records but received {data_count}'
                        logger.error(msg)
                        raise InsufficientDataException(msg)
                else:
                    # Markers have an IRREGULAR_RATE.
                    logger.info(f'Querying {name} data')
                    output[content_type] = client.get_data(start=adjusted_start,
                                                           end=adjusted_start + seconds)
                    logger.info(f"Received {len(output[content_type])} records.")
            else:
                logger.error(f"No client and device spec found for content type: {content_type.name}")
        return output

    def cleanup(self) -> None:  # type: ignore
        """Performs any necessary cleanup tasks for all managed clients."""
        for client in self.clients:
            client.cleanup()

    def __getattr__(self, name: str) -> Any:
        """Dispatches unknown properties and methods to the client with the
        default content type.

        This allows `ClientManager` to act as a proxy for the default client.

        Args:
            name (str): The name of the attribute being accessed.

        Returns:
            Any: The attribute from the default client.

        Raises:
            AttributeError: If the default client is not set or the attribute
                            does not exist on the default client.
        """
        client = self.default_client
        if client:
            return client.__getattribute__(name)

        logger.error(f"Missing attribute: {name}")
        raise AttributeError(f"Missing attribute: {name}")
