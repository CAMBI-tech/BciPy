# pylint: disable=no-name-in-module,missing-docstring,too-few-public-methods
import sys
from pathlib import Path
from typing import Union

from PyQt6 import QtGui
from PyQt6.QtWidgets import QApplication, QFileDialog, QWidget

from bcipy.preferences import preferences
from bcipy.exceptions import BciPyCoreException

DEFAULT_FILE_TYPES = "All Files (*)"


class FileDialog(QWidget):
    """GUI window that prompts the user to select a file."""

    def __init__(self):
        super().__init__()
        self.title = 'File Dialog'
        self.width = 640
        self.height = 480

        # Center on screen
        self.resize(self.width, self.height)
        frame_geom = self.frameGeometry()
        frame_geom.moveCenter(QtGui.QGuiApplication.primaryScreen().availableGeometry().center())
        self.move(frame_geom.topLeft())

        # The native dialog may prevent the selection from closing after a
        # directory is selected.
        self.options = QFileDialog.Option.DontUseNativeDialog

    def ask_file(self,
                 file_types: str = DEFAULT_FILE_TYPES,
                 directory: str = "",
                 prompt: str = "Select File") -> str:
        """Opens a file dialog window.
        Returns
        -------
        path or None
        """
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  caption=prompt,
                                                  directory=directory,
                                                  filter=file_types,
                                                  options=self.options)
        return filename

    def ask_directory(self, directory: str = "", prompt: str = "Select Directory") -> str:
        """Opens a dialog window to select a directory.

        Returns
        -------
        path or None
        """
        return QFileDialog.getExistingDirectory(self,
                                                prompt,
                                                directory=directory,
                                                options=self.options)


def ask_filename(
        file_types: str = DEFAULT_FILE_TYPES,
        directory: str = "",
        prompt: str = "Select File") -> Union[str, BciPyCoreException]:
    """Prompt for a file using a GUI.

    Parameters
    ----------
    - file_types : optional file type filters; Examples: 'Text files (*.txt)'
    or 'Image files (*.jpg *.gif)' or '*.csv;;*.pkl'
    - directory : optional directory

    Returns
    -------
    path to file or raises an exception if the user cancels the dialog.
    """
    app = QApplication(sys.argv)
    dialog = FileDialog()
    directory = directory or preferences.last_directory
    filename = dialog.ask_file(file_types, directory, prompt=prompt)

    # update last directory preference
    path = Path(filename)
    if filename and path.is_file():
        preferences.last_directory = str(path.parent)

        # Alternatively, we could use `app.closeAllWindows()`
        app.quit()

        return filename

    raise BciPyCoreException('No file selected.')


def ask_directory(prompt: str = "Select Directory") -> Union[str, BciPyCoreException]:
    """Prompt for a directory using a GUI.

    Parameters
    ----------
    prompt : optional prompt message to display to users

    Returns
    -------
    path to directory or raises an exception if the user cancels the dialog.
    """
    app = QApplication(sys.argv)

    dialog = FileDialog()
    directory = ''
    if preferences.last_directory:
        directory = str(Path(preferences.last_directory).parent)
    name = dialog.ask_directory(directory, prompt=prompt)
    if name and Path(name).is_dir():
        preferences.last_directory = name

        # Alternatively, we could use `app.closeAllWindows()`
        app.quit()

        return name

    raise BciPyCoreException('No directory selected.')
