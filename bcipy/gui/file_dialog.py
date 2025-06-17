"""File dialog module.

This module provides functionality for displaying file and directory selection
dialogs in the BciPy GUI interface. It includes classes and functions for handling
file and directory selection with customizable options and filters.
"""

# pylint: disable=no-name-in-module,missing-docstring,too-few-public-methods
import sys
from pathlib import Path
from typing import Union, Optional, Tuple

from PyQt6 import QtGui
from PyQt6.QtWidgets import QApplication, QFileDialog, QWidget
from PyQt6.QtCore import QRect

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
        window_width (int): Window width in pixels.
        window_height (int): Window height in pixels.
        options (QFileDialog.Option): Dialog options.
    """

    def __init__(self) -> None:
        """Initialize the file dialog window.

        Sets up the window properties and centers it on the screen.
        """
        super().__init__()
        self.title = 'File Dialog'
        self.window_width = 640
        self.window_height = 480

        # Center on screen
        self.resize(self.window_width, self.window_height)
        self._center_window()

        # The native dialog may prevent the selection from closing after a
        # directory is selected.
        self.options = QFileDialog.Option.DontUseNativeDialog

    def _center_window(self) -> None:
        """Center the window on the primary screen.
        
        This method calculates the center position of the primary screen and
        moves the window to that position.
        """
        frame_geom = self.frameGeometry()
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen:
            center_point = screen.availableGeometry().center()
            frame_geom.moveCenter(center_point)
            self.move(frame_geom.topLeft())

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
        app.quit()
        return name

    if strict:
        raise BciPyCoreException('No directory selected.')

    return ''
