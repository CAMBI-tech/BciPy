"""Multi-modal data acquisition."""

from bcipy.acquisition.client import DataAcquisitionClient
from bcipy.acquisition.protocols.connector import Connector
from bcipy.acquisition.protocols.lsl.lsl_connector import LslConnector, LSL_TIMESTAMP
from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.marker_writer import MarkerWriter
import logging
import time
log = logging.getLogger(__name__)


def validate_connectors(connector: Connector,
                        secondary_connector: Connector) -> None:
    """Validate that the provided connectors are suitable for multi-modal."""
    assert connector.device_spec.content_type == 'EEG', 'An EEG device must be configured'
    if secondary_connector:
        # Both are using LSL connection
        assert connector.__class__.connection_method(
        ) == ConnectionMethod.LSL and secondary_connector.__class__.connection_method(
        ) == ConnectionMethod.LSL
        assert secondary_connector.device_spec.content_type != 'EEG', 'Only one EEG device is allowed'


# TODO: create an abstract interface
class MultiModalDataAcquisitionClient:
    """Coordinates the acquisition of data from multiple sources.
    
    Parameters
    ----------
        connector: LslConnector instance for EEG data
            Object with device-specific implementations for connecting,
            initializing, and reading a packet.
        buffer_name : str, optional
            Name of the sql database archive for EEG data.
        raw_data_file_name: str,
            Name of the raw EEG data csv file to output; if not present raw
                data is not written.
        secondary_connector: LslConnector instance for non-EEG data
            Object with device-specific implementations for connecting,
            initializing, and reading a packet from a secondary data source.
        secondary_buffer_name : str, optional
            Name of the sql database archive for non-EEG data.
        secondary_raw_data_file_name: str,
            Name of the raw non-EEG data csv file to output; if not present raw
            data is not written.
        delete_archive: boolean, optional
            Flag indicating whether to delete the database archive on exit.
            Default is False.
    """

    def __init__(self,
                 connector: Connector,
                 buffer_name: str = 'eeg_raw_data.db',
                 raw_data_file_name: str = 'eeg_raw_data.csv',
                 secondary_connector: LslConnector = None,
                 secondary_buffer_name: str = 'other_raw_data.db',
                 secondary_raw_data_file_name: str = 'other_raw_data.csv',
                 delete_archive: bool = True):
        super().__init__()

        validate_connectors(connector, secondary_connector)

        # Initialize clients
        self.eeg_client = DataAcquisitionClient(
            connector=connector,
            buffer_name=buffer_name,
            raw_data_file_name=raw_data_file_name,
            delete_archive=delete_archive)

        self.secondary_client = None
        if secondary_connector:
            connector.lsl_timestamp_included = True
            secondary_connector.lsl_timestamp_included = True
            self.secondary_client = DataAcquisitionClient(
                connector=secondary_connector,
                buffer_name=secondary_buffer_name,
                raw_data_file_name=secondary_raw_data_file_name,
                delete_archive=delete_archive)

    @property
    def marker_writer(self) -> MarkerWriter:
        return self.eeg_client.marker_writer

    # @override
    def start_acquisition(self) -> None:
        """Run the initialization code to start acquiring data from both data
        sources."""
        start_time = time.time()
        self.eeg_client.start_acquisition()
        log.debug("EEG client startup time: %f", time.time() - start_time)

        if self.secondary_client:
            start_time = time.time()
            self.secondary_client.start_acquisition()
            log.debug("Secondary client startup time: %f",
                      time.time() - start_time)

    def stop_acquisition(self) -> None:
        """Stop acquiring data; perform cleanup."""
        self.eeg_client.stop_acquisition()
        if self.secondary_client:
            self.secondary_client.stop_acquisition()

    def get_data(self, start=None, end=None, field='_rowid_'):
        """Queries the primary data source by field.

        Parameters
        ----------
            start : number, optional
                start of time slice; units are those of the acquisition clock.
            end : float, optional
                end of time slice; units are those of the acquisition clock.
            field: str, optional
                field on which to query; default value is the row id.
        Returns
        -------
            list of Records
        """
        return self.eeg_client.get_data(start, end, field)

    def get_secondary_data(self, start=None, end=None, field='_rowid_'):
        """Get data from the secondary source."""
        if self.secondary_client:
            data = self.get_data(start, end, field)
            if data:
                lsl_timestamp_col = self.eeg_client.device_info.channels.index(
                    LSL_TIMESTAMP)

                start_time = data[0].data[lsl_timestamp_col]
                end_time = data[-1].data[lsl_timestamp_col]
                # TODO: query is returning data with lsl_stamp < end_time, which may exclude some data.
                return self.secondary_client.get_data(start=start_time,
                                                      end=end_time,
                                                      field=LSL_TIMESTAMP)
        return []

    def get_data_len(self):
        """Efficient way to calculate the amount of data cached."""
        return self.eeg_client.get_data_len()

    @property
    def device_info(self):
        return self.eeg_client.device_info

    @property
    def is_calibrated(self):
        """Returns boolean indicating whether or not acquisition has been
        calibrated (an offset calculated based on a trigger)."""
        return self.eeg_client.is_calibrated

    @is_calibrated.setter
    def is_calibrated(self, bool_val):
        """Setter for the is_calibrated property that allows the user to
        override the calculated value and use a 0 offset.

        Parameters
        ----------
            bool_val: boolean
                if True, uses a 0 offset; if False forces the calculation.
        """
        self.eeg_client.is_calibrated = bool_val

    @property
    def offset(self):
        return self.eeg_client.offset

    def cleanup(self):
        self.eeg_client.cleanup()
        if self.secondary_client:
            self.secondary_client.cleanup()