# pylint: disable=no-name-in-module,missing-docstring,too-few-public-methods
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QDesktopWidget

DEFAULT_FILE_TYPES = "All Files (*)"


class FileDialog(QWidget):
    """GUI window that prompts the user to select a file."""

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.width = 640
        self.height = 480

        # Center on screen
        self.resize(self.width, self.height)
        frame_geom = self.frameGeometry()
        frame_geom.moveCenter(QDesktopWidget().availableGeometry().center())
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

    def ask_directory(self) -> str:
        """Opens a dialog window to select a directory.

        Returns
        -------
        path or None
        """
        return QFileDialog.getExistingDirectory(self,
                                                "Select Directory",
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
    filename = dialog.ask_file(file_types, directory)

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
    name = dialog.ask_directory()

    # Alternatively, we could use `app.closeAllWindows()`
    app.quit()

    return name
