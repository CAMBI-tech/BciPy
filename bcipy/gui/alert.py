"""GUI alert messages module.

This module provides functionality for displaying alert messages and confirmation dialogs
in the BciPy GUI interface. It includes functions for user interaction through
standard dialog boxes with customizable options.
"""

# pylint: disable=no-name-in-module
import sys
from typing import Optional

from PyQt6.QtWidgets import QApplication

from bcipy.gui.main import (AlertMessageResponse, AlertMessageType,
                            AlertResponse, alert_message)


def confirm(message: str) -> bool:
    """Confirmation dialog which allows the user to select between a true and false.

    This function displays a dialog box with OK and Cancel buttons, allowing the user
    to confirm or cancel an action. The dialog is displayed using the system's native
    dialog style.

    Args:
        message (str): The alert message to display in the dialog box.

    Returns:
        bool: True if the user clicked OK, False if the user clicked Cancel.

    Note:
        This function creates a new QApplication instance if one doesn't exist,
        and quits it after the dialog is closed.
    """
    app = QApplication(sys.argv).instance()
    if not app:
        app = QApplication(sys.argv)
    dialog = alert_message(message,
                           message_type=AlertMessageType.INFO,
                           message_response=AlertMessageResponse.OCE)
    button = dialog.exec()
    result = bool(button == AlertResponse.OK.value)
    app.quit()
    return result
