"""GUI alert messages"""
# pylint: disable=no-name-in-module
import sys
from PyQt5.QtWidgets import QApplication, QMessageBox


def confirm(message: str) -> bool:
    """Confirmation dialog which allows the user to select between a true and false.

    Parameters
    ----------
        message - alert to display
    Returns
    -------
        users selection
    """
    app = QApplication(sys.argv)
    dialog = QMessageBox()
    dialog.setText(message)
    dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    dialog.setIcon(QMessageBox.Information)
    button = dialog.exec()

    result = bool(button == QMessageBox.Ok)
    app.quit()
    return result
