# pylint: disable=no-name-in-module,missing-docstring,too-few-public-methods
import sys
from pathlib import Path

from PyQt6 import QtGui
from PyQt6.QtWidgets import QApplication, QFileDialog, QWidget

from bcipy.preferences import preferences

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
        prompt: str = "Select File") -> str:
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
    filename = dialog.ask_file(file_types, directory, prompt=prompt)

    # update last directory preference
    path = Path(filename)
    if filename and path.is_file():
        preferences.last_directory = str(path.parent)

    # Alternatively, we could use `app.closeAllWindows()`
    app.quit()

    return filename


def ask_directory(prompt: str = "Select Directory") -> str:
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
    name = dialog.ask_directory(directory, prompt=prompt)
    if name and Path(name).is_dir():
        preferences.last_directory = name

    # Alternatively, we could use `app.closeAllWindows()`
    app.quit()

    return name
