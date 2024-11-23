"""Graphical User Interface for configuring the Simulator."""

# pylint: disable=E0611
import argparse
import fnmatch
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QCheckBox, QFormLayout, QHBoxLayout,
                             QLineEdit, QPushButton, QScrollArea, QSpinBox,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.gui.file_dialog import FileDialog
from bcipy.gui.main import static_text_control
from bcipy.helpers.acquisition import active_content_types
from bcipy.helpers.parameters import Parameters
from bcipy.preferences import preferences
from bcipy.simulator.ui.cli import excluded
from bcipy.simulator.util.artifact import DEFAULT_SAVE_LOCATION


def walk_directory(directory: Path,
                   tree: Union[QTreeWidget, QTreeWidgetItem],
                   skip_dir: Callable[[Path], bool] = lambda p: False) -> None:
    """Recursively build a Tree with directory contents.
    Adapted from rich library examples/tree.py

    Parameters
    ----------
        directory - root directory of the tree
        tree - Tree object that gets recursively updated
        skip_dir - predicate to determine if a directory should be skipped
    """
    # Sort dirs first then by filename
    paths = sorted(
        Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    for path in paths:
        # Remove hidden files
        if path.name.startswith(".") or path.is_file() or (path.is_dir()
                                                           and skip_dir(path)):
            continue
        if path.is_dir():
            branch = QTreeWidgetItem(tree, [path.name])
            walk_directory(path, branch, skip_dir)


def is_ancestor(directory: Path, paths: List[Path]) -> bool:
    """Check if the given directory is the ancestor of any of the provided paths."""
    for path in paths:
        if directory in path.parents:
            return True
    return False


class DirectoryTree(QWidget):
    """Display a tree of directories"""

    def __init__(self,
                 parent_directory: Optional[str] = None,
                 selected_subdirectories: Optional[List[str]] = None):
        super().__init__()
        self.parent_directory = parent_directory
        self.paths = selected_subdirectories or []

        self.label = "Selected Data Dictionaries"
        self.tree = self.make_tree()
        self.layout = QVBoxLayout()
        self.layout.addWidget(LabeledWidget(self.label, self.tree))
        self.layout.addWidget(self.tree)
        self.setLayout(self.layout)

    def update(self, parent_directory: str,
               selected_subdirectories: List[str]) -> None:
        """Update the widget."""
        self.parent_directory = parent_directory
        self.paths = selected_subdirectories or []

        clear_layout(self.layout)
        if self.parent_directory:
            self.tree = self.make_tree()
            self.layout.addWidget(LabeledWidget(self.label, self.tree))

    def make_tree(self) -> QTreeWidget:
        """Initialize the tree widget"""

        tree = QTreeWidget()
        if self.parent_directory:
            tree.setColumnCount(1)
            tree.setHeaderLabels([self.parent_directory])
            walk_directory(Path(self.parent_directory),
                           tree,
                           skip_dir=self.skip_path)
        return tree

    def skip_path(self, directory: Path) -> bool:
        """Should the given directory be skipped"""
        if directory in self.paths or is_ancestor(directory, self.paths):
            return False
        return True


class ChooseFileInput(QWidget):
    """Text field and button which pops open a file selection dialog."""

    def __init__(self,
                 value: Optional[str] = None,
                 file_selector: str = "*",
                 prompt: str = "Select a file",
                 change_event: Optional[Callable] = None):

        super().__init__()
        self.file_selector = file_selector
        self.prompt = prompt
        self.change_event = change_event
        self.control = QLineEdit(value)
        self.control.textChanged.connect(self.change)

        btn = QPushButton('...')
        btn.setFixedWidth(40)
        btn.clicked.connect(self.update_path)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.control)
        hbox.addWidget(btn)
        self.setLayout(hbox)

    def prompt_path(self) -> str:
        """Prompts the user with a FileDialog. Returns the selected value or None."""
        dialog = FileDialog()
        directory = preferences.last_directory
        filename = dialog.ask_file(self.file_selector,
                                   directory,
                                   prompt=self.prompt)
        path = Path(filename)
        if filename and path.is_file():
            preferences.last_directory = str(path.parent)
        return filename

    def update_path(self) -> None:
        """Prompts the user for a value and updates the control's value if one was input."""
        name = self.prompt_path()
        if name:
            self.control.setText(name)

    def change(self):
        """Called when the path changes"""
        if self.change_event:
            self.change_event()

    def value(self) -> Optional[str]:
        """Path value"""
        return self.control.text()


class ChooseDirectoryInput(ChooseFileInput):
    """Text field and button which pops open a directory selection dialog."""

    def __init__(self,
                 value: Optional[str] = None,
                 prompt: str = "Select a directory",
                 change_event: Optional[Callable] = None):
        super().__init__(value=value, prompt=prompt, change_event=change_event)

    def prompt_path(self):
        dialog = FileDialog()
        directory = ''
        if preferences.last_directory:
            directory = str(Path(preferences.last_directory).parent)
        name = dialog.ask_directory(directory, prompt=self.prompt)
        if name and Path(name).is_dir():
            preferences.last_directory = name
        return name


class ParameterFileInput(ChooseFileInput):
    """Prompts for a parameters file."""

    def __init__(self, **kwargs):
        kwargs['file_selector'] = "*.json"
        kwargs['prompt'] = "Select a parameters file"
        super().__init__(**kwargs)
        self.parameters: Optional[Parameters] = None

    def change(self):
        """Connected to a change in the user input parameters."""
        value = self.value()
        if value is not None and Path(value).is_file():
            self.parameters = Parameters(value, cast_values=True)
        else:
            self.parameters = None
        if self.change_event:
            self.change_event()


class DirectoryFilters(QWidget):
    """Fields that filter the selected directories"""

    def __init__(self, change_event: Optional[Callable], **kwargs):
        super().__init__(**kwargs)
        self.change_event = change_event
        self.layout = QVBoxLayout()

        self.nested_filter_control = QCheckBox("Include nested directories")
        self.nested_filter_control.setChecked(True)
        self.nested_filter_control.checkStateChanged.connect(self.change)

        # self.name_filter_control_label = QLabel("Name contains")
        self.name_filter_control = QLineEdit("")
        self.name_filter_control.textChanged.connect(self.change)
        self.layout.addWidget(self.nested_filter_control)

        form = QFormLayout()
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.addRow("Name contains", self.name_filter_control)

        self.layout.addLayout(form)
        self.setLayout(self.layout)

    def change(self):
        """Called when a filter field is modified."""
        if self.change_event:
            self.change_event()


class DataDirectorySelect(QWidget):
    """Widget that facilitates the selection of data directories.
    Includes an input for a parent directory and optional filters.
    """

    def __init__(self,
                 parent_directory: Optional[str] = None,
                 change_event: Optional[Callable] = None):
        super().__init__()
        self.layout = QVBoxLayout()
        self.change_event = change_event

        self.parent_directory_control = ChooseDirectoryInput(
            change_event=self.update_directory_tree,
            prompt="Select a parent data directory",
            value=parent_directory)

        self.directory_filter_control = DirectoryFilters(
            change_event=self.update_directory_tree)
        self.directory_tree = DirectoryTree()

        self.parent_directory_control.layout().setContentsMargins(0, 0, 0, 0)
        self.directory_filter_control.layout.setContentsMargins(0, 0, 0, 0)
        # self.directory_filter_control.setVisible(False)
        # self.directory_tree.setVisible(False)

        self.layout.addWidget(
            LabeledWidget("Data Parent Directory",
                          self.parent_directory_control))
        self.layout.addWidget(self.directory_filter_control)
        self.layout.addWidget(self.directory_tree)

        self.setLayout(self.layout)

    def parent_directory(self) -> Optional[str]:
        """Parent directory"""
        parent = self.parent_directory_control.value()
        if parent:
            return parent.strip()

    def match_pattern(self) -> str:
        """Pattern used to match a directory with fnmatch."""
        name_filter = self.directory_filter_control.name_filter_control.text()
        if name_filter:
            return f"*{name_filter.strip(' *')}*"
        return "*"

    def use_nested(self) -> bool:
        """Include nested directories"""
        return self.directory_filter_control.nested_filter_control.isChecked()

    def data_directories(self) -> Optional[List[Path]]:
        """Compute the data directories of interest."""
        parent = self.parent_directory()
        if not parent:
            return None

        pattern = self.match_pattern()
        subdirs = []
        if self.use_nested():
            for cur_path, directories, _files in os.walk(parent, topdown=True):
                for name in fnmatch.filter(directories, pattern):
                    path = Path(cur_path, name)
                    if not excluded(path):
                        subdirs.append(path)
        else:
            cur_path, directories, _files = next(os.walk(parent))
            for name in fnmatch.filter(directories, pattern):
                path = Path(cur_path, name)
                if not excluded(path):
                    subdirs.append(path)
        return sorted(subdirs)

    def update_directory_tree(self):
        """Update the directory tree"""
        # self.directory_filter_control.setVisible(False)
        # self.directory_tree.setVisible(False)
        if self.parent_directory():
            # self.directory_filter_control.setVisible(True)
            # self.directory_tree.setVisible(True)
            self.directory_tree.update(self.parent_directory(),
                                       self.data_directories())
        # notify any listeners of the change
        if self.change_event:
            self.change_event()


class ModelFileInput(ChooseFileInput):
    """Prompts for a parameters file."""

    def __init__(self,
                 value: Optional[str] = None,
                 file_selector: str = "*.pkl",
                 prompt: str = "Select a model",
                 change_event: Optional[Callable] = None):
        super().__init__(value=value,
                         file_selector=file_selector,
                         prompt=prompt,
                         change_event=change_event)


class ModelInputs(QWidget):
    """Provide a model path input for each configured content type."""

    def __init__(self, content_types: Optional[List[str]] = None):
        super().__init__()

        self.layout = QFormLayout()
        self.layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft
                                     | Qt.AlignmentFlag.AlignVCenter)
        self.layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.controls = self._create_inputs(content_types or ['EEG'])
        self._add_controls()
        self.setLayout(self.layout)

    def set_content_types(self, value: List[str]):
        """Set the content_content types and re-generate inputs."""
        self.controls = self._create_inputs(value)
        self._add_controls()

    def _create_inputs(self,
                       content_types: List[str]) -> List[Tuple[str, QWidget]]:
        """Create a path input for each model based on the configured acq_mode."""
        return [(f"{content_type} Model", ModelFileInput())
                for content_type in content_types]

    def _add_controls(self) -> None:
        """Add each model input to the layout."""
        clear_layout(self.layout)
        for (label, control) in self.controls:
            self.layout.addRow(label, control)


class LabeledWidget(QWidget):
    """Renders a widget with a label above it."""

    def __init__(self, label: str, widget: QWidget, label_size: int = 14):
        super().__init__()
        vbox = QVBoxLayout()
        label = static_text_control(None, label, size=label_size)
        label.setMargin(0)
        vbox.setSpacing(0)
        vbox.setContentsMargins(-1, 0, -1, -1)
        vbox.addWidget(label)
        vbox.addWidget(widget)
        self.setLayout(vbox)


def sim_runs_control(value: int = 1) -> QWidget:
    """Create a widget for entering simulation runs."""
    # TODO: event
    spin_box = QSpinBox()
    spin_box.setMinimum(1)
    spin_box.setMaximum(1000)
    spin_box.wheelEvent = lambda event: None  # disable scroll wheel
    spin_box.setValue(value)
    return spin_box


class SimConfigForm(QWidget):
    """The InputForm class is a QWidget that creates controls/inputs for each simulation parameter

  Parameters:
  -----------
    json_file - path of parameters file to be edited.
    width - optional; used to set the width of the form controls.
  """

    def __init__(self, width: int = 400):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        # self.setFixedWidth(width)

        self.parameter_control = ParameterFileInput(
            change_event=self.update_parameters)

        self.directory_control = DataDirectorySelect()
        self.model_input_control = ModelInputs()
        self.runs_control = sim_runs_control()
        self.output_control = ChooseDirectoryInput(value=DEFAULT_SAVE_LOCATION)

        form = QFormLayout()
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft
                              | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        select_file = ChooseFileInput(file_selector="*.json",
                                      prompt="Select a parameters file")
        form.addRow("Simulation Runs:", sim_runs_control())
        form.addRow("Output Directory:", self.output_control)
        form.addRow("Sim Parameters:", self.parameter_control)

        # self.layout.addWidget(
        #     static_text_control(None, "Simulation Parameters", size=14))
        # self.layout.addWidget(LabeledWidget("Simulation Parameters", self.parameter_control))

        self.layout.addLayout(form)
        self.layout.addWidget(self.model_input_control)
        self.layout.addWidget(self.directory_control)
        # self.layout.addWidget(self.output_control)
        # self.layout.addWidget(self.runs_control)

        # self.layout.setSpacing(0)
        # self.layout.setContentsMargins(0, 0, 0, 0)
        self.show()

    @property
    def parameters(self) -> Optional[Parameters]:
        """Configured parameters"""
        return self.parameter_control.parameters

    def command_valid(self) -> bool:
        """Returns True if all necessary fields are input."""
        # TODO:
        return False

    def command(self) -> str:
        """Command equivalent to to the result of the interactive selection of
        simulator inputs."""
        params = ''
        models = []
        source_dirs = []

        model_args = ' '.join([f"-m {path}" for path in models])
        dir_args = ' '.join(f"-d {source}" for source in source_dirs)
        return f"bcipy-sim -p {params} {model_args} {dir_args}"

    def update_parameters(self) -> None:
        """When simulation parameters are updated include an input for each model."""
        print("Updating parameters")
        params = self.parameter_control.parameters
        if params:
            content_types = active_content_types(params.get('acq_mode', 'EEG'))
        else:
            content_types = ['EEG']
        print("content types:")
        print(content_types)
        self.model_input_control.set_content_types(content_types)


def clear_layout(layout):
    """Clear all items from a layout.
    https://stackoverflow.com/questions/9374063/remove-all-items-from-a-layout
    """
    while layout.count():
        child = layout.takeAt(0)
        if child.widget() is not None:
            child.widget().setVisible(False)
            child.widget().deleteLater()
        elif child.layout() is not None:
            clear_layout(child.layout())


class MainPanel(QWidget):
    """Main GUI window.
    Parameters:
    -----------
      json_file: path to the parameters file.
      title: window title
      size: window size; (width, height)
    """

    def __init__(self, title: str, size: Tuple[int, int]):
        super().__init__()
        self.title = title
        self.size = size

        self.command = ''

        self.form = SimConfigForm(width=self.size[0])
        self.flash_msg = ''
        self.init_ui()

    def init_ui(self):
        """Initialize the UI"""
        vbox = QVBoxLayout()

        header_box = QHBoxLayout()
        header_box.addSpacing(4)
        header_box.addWidget(
            static_text_control(self,
                                label='BciPy Simulator Configuration',
                                size=16,
                                color='dimgray'))
        vbox.addLayout(header_box)

        self.form_panel = QScrollArea()
        self.form_panel.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.form_panel.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.form_panel.setWidgetResizable(True)
        self.form_panel.setMinimumWidth(self.size[0])
        self.form_panel.setWidget(self.form)

        vbox.addWidget(self.form_panel)
        vbox.addSpacing(4)

        # Controls
        control_box = QHBoxLayout()
        control_box.addStretch()
        run_button = QPushButton('Run')
        run_button.setFixedWidth(80)
        run_button.clicked.connect(self.on_run)
        control_box.addWidget(run_button)

        vbox.addLayout(control_box)

        self.command_msg = static_text_control(self,
                                               label='',
                                               size=12,
                                               color='darkgray')
        vbox.addWidget(self.command_msg)
        self.setLayout(vbox)
        self.setFixedHeight(self.size[1])
        self.setWindowTitle(self.title)
        self.show()

    def on_run(self):
        """Event handler for running the simulation."""

        if self.form.command_valid():
            self.command_msg.setText(self.form.command())
            self.command_msg.repaint()


def init(title='BCI Simulator', size=(750, 600)) -> str:
    """Set up the GUI components and start the main loop."""
    app = QApplication(sys.argv)
    panel = MainPanel(title, size)
    app.exec()
    command = panel.command
    app.quit()
    return command


def main():
    """Process command line arguments and initialize the GUI."""
    parser = argparse.ArgumentParser()

    # Command line utility for adding arguments/ paths via command line
    parser.add_argument('-p',
                        '--parameters',
                        default=DEFAULT_PARAMETERS_PATH,
                        help='Path to parameters.json configuration file.')
    args = parser.parse_args()
    # Note that this write to stdout is important for the interaction with
    # the BCInterface main GUI.
    print(init(args.parameters), file=sys.stdout)


if __name__ == '__main__':
    main()
