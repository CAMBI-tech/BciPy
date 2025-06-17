"""File dialog module.

This module provides functionality for displaying file and directory selection
dialogs in the BciPy GUI interface.
"""

# pylint: disable=no-name-in-module,missing-docstring,too-few-public-methods
import sys
from pathlib import Path
from typing import Union, Optional

from PyQt6 import QtGui
from PyQt6.QtWidgets import QApplication, QFileDialog, QWidget

from bcipy.exceptions import BciPyCoreException
from bcipy.preferences import preferences

DEFAULT_FILE_TYPES: str = "All Files (*)"


class FileDialog(QWidget):
    """GUI window that prompts the user to select a file or directory.

    This class provides a file dialog interface for selecting files and directories
    in the BciPy GUI. It supports both file and directory selection with customizable
    options and filters.

    Attributes:
        title (str): Window title.
        width (int): Window width in pixels.
        height (int): Window height in pixels.
        options (QFileDialog.Option): Dialog options.
    """

    def __init__(self) -> None:
        """Initialize the file dialog window.

        Sets up the window properties and centers it on the screen.
        """
        super().__init__()
        self.title = 'File Dialog'
        self.width: int = 640
        self.height: int = 480

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
        """Open a file selection dialog window.

        Args:
            file_types (str, optional): File type filters. Defaults to DEFAULT_FILE_TYPES.
            directory (str, optional): Initial directory. Defaults to "".
            prompt (str, optional): Dialog prompt message. Defaults to "Select File".

        Returns:
            str: Selected file path or empty string if cancelled.
        """
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  caption=prompt,
                                                  directory=directory,
                                                  filter=file_types,
                                                  options=self.options)
        return filename

    def ask_directory(self, directory: str = "", prompt: str = "Select Directory") -> str:
        """Open a directory selection dialog window.

        Args:
            directory (str, optional): Initial directory. Defaults to "".
            prompt (str, optional): Dialog prompt message. Defaults to "Select Directory".

        Returns:
            str: Selected directory path or empty string if cancelled.
        """
        return QFileDialog.getExistingDirectory(self,
                                                prompt,
                                                directory=directory,
                                                options=self.options)


def ask_filename(
        file_types: str = DEFAULT_FILE_TYPES,
        directory: str = "",
        prompt: str = "Select File",
        strict: bool = False) -> Union[str, BciPyCoreException]:
    """Prompt for a file using a GUI.

    This function creates a file selection dialog and handles the user's selection.
    It can optionally raise an exception if no file is selected.

    Args:
        file_types (str, optional): File type filters. Examples: 'Text files (*.txt)'
            or 'Image files (*.jpg *.gif)' or '*.csv;;*.pkl'. Defaults to DEFAULT_FILE_TYPES.
        directory (str, optional): Initial directory. Defaults to "".
        prompt (str, optional): Dialog prompt message. Defaults to "Select File".
        strict (bool, optional): If True, raises an exception when no file is selected.
            If False, returns an empty string. Defaults to False.

    Returns:
        Union[str, BciPyCoreException]: Selected file path or empty string if cancelled.
            Raises BciPyCoreException if strict=True and no file is selected.

    Note:
        Updates the last_directory preference if a file is selected.
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

    if strict:
        raise BciPyCoreException('No file selected.')

    return ''


def ask_directory(prompt: str = "Select Directory", strict: bool = False) -> Union[str, BciPyCoreException]:
    """Prompt for a directory using a GUI.

    This function creates a directory selection dialog and handles the user's selection.
    It can optionally raise an exception if no directory is selected.

    Args:
        prompt (str, optional): Dialog prompt message. Defaults to "Select Directory".
        strict (bool, optional): If True, raises an exception when no directory is selected.
            If False, returns an empty string. Defaults to False.

    Returns:
        Union[str, BciPyCoreException]: Selected directory path or empty string if cancelled.
            Raises BciPyCoreException if strict=True and no directory is selected.

    Note:
        Updates the last_directory preference if a directory is selected.
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

    if strict:
        raise BciPyCoreException('No directory selected.')

    return ''
