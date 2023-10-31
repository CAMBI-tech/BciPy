# pylint: disable=no-name-in-module,missing-docstring,too-few-public-methods
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt6 import QtGui
from bcipy.preferences import preferences

DEFAULT_FILE_TYPES = "All Files (*)"


class FileDialog(QWidget):
    """GUI window that prompts the user to select a file."""

    def __init__(self):
        super().__init__()
        self.title = 'PyQt6 file dialogs - pythonspot.com'
        self.width = 640
        self.height = 480

        # Center on screen
        self.resize(self.width, self.height)
        frame_geom = self.frameGeometry()
        frame_geom.moveCenter(QtGui.QGuiApplication.primaryScreen().availableGeometry().center())
        self.move(frame_geom.topLeft())

        # The native dialog may prevent the selection from closing after a
        # directory is selected.
        self.options = QFileDialog.Options()
        self.options |= QFileDialog.DontUseNativeDialog

    def ask_file(self, file_types: str = DEFAULT_FILE_TYPES, directory: str = "") -> str:
        """Opens a file dialog window.
        Returns
        -------
        path or None
        """
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Select File",
                                                  directory,
                                                  file_types,
                                                  options=self.options)
        return filename

    def ask_directory(self, directory: str = "") -> str:
        """Opens a dialog window to select a directory.

        Returns
        -------
        path or None
        """
        return QFileDialog.getExistingDirectory(self,
                                                "Select Directory",
                                                directory=directory,
                                                options=self.options)


def ask_filename(file_types: str = DEFAULT_FILE_TYPES, directory: str = "") -> str:
    """Prompt for a file.

    Parameters
    ----------
    - file_types : optional file type filters; Examples: 'Text files (*.txt)'
    or 'Image files (*.jpg *.gif)' or '*.csv;;*.pkl'
    - directory : optional directory

    Returns
    -------
    path to file or None if the user cancelled the dialog.
    """
    app = QApplication(sys.argv)
    dialog = FileDialog()
    directory = directory or preferences.last_directory
    filename = dialog.ask_file(file_types, directory)

    # update last directory preference
    path = Path(filename)
    if filename and path.is_file():
        preferences.last_directory = str(path.parent)

    # Alternatively, we could use `app.closeAllWindows()`
    app.quit()

    return filename


def ask_directory() -> str:
    """Prompt for a directory.

    Returns
    -------
    path to directory or None if the user cancelled the dialog.
    """
    app = QApplication(sys.argv)

    dialog = FileDialog()
    directory = ''
    if preferences.last_directory:
        directory = str(Path(preferences.last_directory).parent)
    name = dialog.ask_directory(directory)
    if name and Path(name).is_dir():
        preferences.last_directory = name
    # Alternatively, we could use `app.closeAllWindows()`
    app.quit()

    return name
