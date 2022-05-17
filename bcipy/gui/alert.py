"""GUI alert messages"""
# pylint: disable=no-name-in-module
import sys
from PyQt5.QtWidgets import QApplication
from bcipy.gui.main import alert_message, AlertMessageType, AlertResponse, AlertMessageResponse


def confirm(message: str) -> bool:
    """Confirmation dialog which allows the user to select between a true and false.

    Parameters
    ----------
        message - alert to display
    Returns
    -------
        users selection : True for selecting Ok, False for Cancel.
    """
    app = QApplication(sys.argv)
    dialog = alert_message(message,
                           message_type=AlertMessageType.INFO,
                           message_response=AlertMessageResponse.OCE)
    button = dialog.exec()

    result = bool(button == AlertResponse.OK.value)
    app.quit()
    return result
