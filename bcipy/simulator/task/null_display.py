"""Display that doesn't do anything"""
from typing import List, Tuple

from bcipy.display import Display


class NullDisplay(Display):
    """Display that doesn't show anything to the user. Useful in simulated tasks
    that do not have a display component."""

    def do_inquiry(self) -> List[Tuple[str, float]]:
        return []

    def wait_screen(self, *args, **kwargs) -> None:
        """Do nothing"""

    def update_task_bar(self, *args, **kwargs) -> None:
        """Do nothing"""
