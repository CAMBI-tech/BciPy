"""Display that doesn't do anything"""
from typing import List, Tuple

from bcipy.acquisition.multimodal import ClientManager


class NullDAQ(ClientManager):
    """Display that doesn't show anything to the user. Useful in simulated tasks
    that do not have a display component."""

    def start_acquisition(self) -> None:
        """Do nothing"""

    def stop_acquisition(self) -> None:
        """Do nothing"""

    def cleanup(self):
        """Do nothing"""

    def get_data_by_device(self, *args, **kwargs) -> None:
        """Do nothing"""
