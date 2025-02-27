"""DAQ that doesn't do anything"""
from bcipy.acquisition.multimodal import ClientManager


class NullDAQ(ClientManager):
    """DAQ that doesn't do anything."""

    def start_acquisition(self) -> None:
        """Do nothing"""

    def stop_acquisition(self) -> None:
        """Do nothing"""

    def cleanup(self) -> None:
        """Do nothing"""

    def get_data_by_device(self, *args, **kwargs) -> None:
        """Do nothing"""
