"""Graphical User Interface for configuring the Simulator."""

# pylint: disable=E0611
import argparse
import fnmatch
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QScrollArea,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.gui.file_dialog import FileDialog
from bcipy.gui.main import (DirectoryInput, FileInput, IntegerInput,
                            static_text_control)
from bcipy.helpers.acquisition import active_content_types
from bcipy.helpers.parameters import Parameters
from bcipy.preferences import preferences
from bcipy.simulator.ui.cli import excluded


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

        self.tree = self.make_tree()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tree)
        self.setLayout(self.layout)
        self.hide()

    def update(self, parent_directory: str,
               selected_subdirectories: List[str]) -> None:
        """Update the widget."""
        self.parent_directory = parent_directory
        self.paths = selected_subdirectories or []

        self.layout.removeWidget(self.tree)
        if self.parent_directory and self.paths:
            self.tree = self.make_tree()
            self.layout.addWidget(self.tree)
            self.show()
        else:
            self.hide()

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

    def show(self):
        """Show this widget, and all child widgets."""
        self.tree.setVisible(True)

    def hide(self):
        """Hide this widget, and all child widgets."""
        self.tree.setVisible(False)


class ParameterFileInput(FileInput):
    """Prompts for a parameters file."""

    def __init__(self, param_change_event: Callable, **kwargs):
        super().__init__(**kwargs)
        self.parameters: Optional[Parameters] = None
        self.param_change_event = param_change_event
        self.control.textChanged.connect(self.on_parameter_change)

    def prompt_path(self) -> str:
        """Prompts the user with a FileDialog. Returns the selected value or None."""
        dialog = FileDialog()
        directory = preferences.last_directory
        filename = dialog.ask_file("*.json",
                                   directory,
                                   prompt="Select a parameters file")

        # update last directory preference
        path = Path(filename)
        if filename and path.is_file():
            preferences.last_directory = str(path.parent)
        return filename

    def on_parameter_change(self):
        """Connected to a change in the user input parameters."""
        if self.value() is not None and Path(self.value()).is_file():
            self.parameters = Parameters(self.value(), cast_values=True)
        else:
            self.parameters = None
        self.param_change_event()


class ParentDirectoryInput(DirectoryInput):
    """Input for the parent directory. Notifies on change."""

    def __init__(self, change_event: Optional[Callable], **kwargs):
        super().__init__(**kwargs)
        self.change_event = change_event
        self.control.textChanged.connect(self.change)

    def prompt_path(self):
        dialog = FileDialog()
        directory = ''
        if preferences.last_directory:
            directory = str(Path(preferences.last_directory).parent)
        name = dialog.ask_directory(directory,
                                    prompt="Select a parent data directory")
        if name and Path(name).is_dir():
            preferences.last_directory = name
        return name

    def separator(self) -> QWidget:
        line = QLabel()
        line.setFixedHeight(0)
        return line

    def change(self):
        if self.change_event:
            self.change_event()


class DataDirectorySelect(QWidget):
    """Widget that facilitates the selection of data directories.
    Includes an input for a parent directory and optional filters.
    """

    def __init__(self, parent_directory: Optional[str] = None):
        super().__init__()
        self.layout = QVBoxLayout()

        self.parent_directory_control = ParentDirectoryInput(
            change_event=self.update_directory_tree,
            label="Data Parent Directory",
            value=parent_directory,
            help_tip="Parent directory for data folders",
            editable=None)

        self.nested_filter_control = QCheckBox("Include nested directories")
        self.nested_filter_control.setChecked(True)
        self.nested_filter_control.checkStateChanged.connect(
            self.update_directory_tree)

        self.name_filter_control_label = QLabel("Name contains")
        self.name_filter_control = QLineEdit("")
        self.name_filter_control.textChanged.connect(
            self.update_directory_tree)

        self.directory_tree = DirectoryTree()

        self.layout.addWidget(self.parent_directory_control)

        self.layout.addWidget(self.nested_filter_control)
        self.layout.addWidget(self.name_filter_control_label)
        self.layout.addWidget(self.name_filter_control)
        self.layout.addWidget(
            static_text_control(self, label="Selected Data Directories"))
        self.layout.addWidget(self.directory_tree)

        self.setLayout(self.layout)

    def parent_directory(self) -> Optional[str]:
        """Parent directory"""
        parent = self.parent_directory_control.value()
        if parent:
            return parent.strip()

    def match_pattern(self) -> str:
        """Pattern used to match a directory with fnmatch."""
        if self.name_filter_control.text():
            return f"*{self.name_filter_control.text().strip(' *')}*"
        return "*"

    def data_directories(self) -> Optional[List[Path]]:
        """Compute the data directories of interest."""
        parent = self.parent_directory()
        if not parent:
            return None

        pattern = self.match_pattern()
        subdirs = []
        if self.nested_filter_control.isChecked():
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
        self.directory_tree.update(self.parent_directory(),
                                   self.data_directories())


class ModelFileInput(FileInput):
    """Prompts for a parameters file."""

    def prompt_path(self) -> str:
        """Prompts the user with a FileDialog. Returns the selected value or None."""
        name, _ = QFileDialog.getOpenFileName(caption='Select a model',
                                              filter='*.pkl')
        return name


class ModelInputs(QWidget):
    """Provide a model path input for each configured content type."""

    def __init__(self, content_types: Optional[List[str]] = None):
        super().__init__()

        self.layout = QVBoxLayout()
        self.controls = self._create_inputs(content_types or ['EEG'])
        self._add_controls()
        self.setLayout(self.layout)

    def set_content_types(self, value: List[str]):
        """Set the content_content types and re-generate inputs."""
        self.controls = self._create_inputs(value)
        self._add_controls()

    def _create_inputs(self, content_types: List[str]) -> List[QWidget]:
        """Create a path input for each model based on the configured acq_mode."""
        return [
            ModelFileInput(label=f"{content_type} Model",
                           value=None,
                           help_tip=f"Path to the {content_type} model.",
                           editable=None) for content_type in content_types
        ]

    def _add_controls(self) -> None:
        """Add each model input to the layout."""
        clear_layout(self.layout)
        for control in self.controls:
            self.layout.addWidget(control)


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
        self.setFixedWidth(width)

        self.parameter_control = ParameterFileInput(
            param_change_event=self.update_parameters,
            label="Parameters",
            value=None,
            help_tip="Path to the simulation parameters.json file.",
            editable=None)

        self.directory_control = DataDirectorySelect()
        self.model_input_control = ModelInputs()
        self.runs_control = IntegerInput(label='Simulation Runs',
                                         value=1,
                                         editable=None)

        self.layout.addWidget(self.parameter_control)
        self.layout.addWidget(self.model_input_control)
        self.layout.addWidget(self.directory_control)
        self.layout.addWidget(self.runs_control)
        self.show()

    @property
    def parameters(self) -> Optional[Parameters]:
        """Configured parameters"""
        return self.parameter_control.parameters

    def command(self, params: str, models: List[str],
                source_dirs: List[str]) -> None:
        """Command equivalent to to the result of the interactive selection of
        simulator inputs."""

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


def do_to_widgets(layout, action):
    """Perform some action on all widgets, including those nested in
    child layouts."""
    for i in range(layout.count()):
        child = layout.itemAt(i)
        if child.widget() is not None:
            action(child.widget())
        elif child.layout() is not None:
            do_to_widgets(child.layout(), action)


def show_all(layout):
    """Show all items in a layout"""
    do_to_widgets(layout, lambda widget: widget.setVisible(True))


def hide_all(layout):
    """Hide all items in a layout"""
    do_to_widgets(layout, lambda widget: widget.setVisible(False))


# class ChangeItems(QWidget):
#     """Line items showing parameter changes."""

#     def __init__(self, json_file: str):
#         super().__init__()
#         self.changes = changes_from_default(json_file)
#         self.layout = QVBoxLayout()
#         self._init_changes()
#         self.setLayout(self.layout)

#     def _init_changes(self):
#         """Initialize the line items for changes."""
#         if not self.changes:
#             self.layout.addWidget(
#                 static_text_control(None,
#                                     label="None",
#                                     size=12,
#                                     color='darkgray'))

#         for _key, param_change in self.changes.items():
#             param = param_change.parameter
#             hbox = QHBoxLayout()

#             lbl = static_text_control(
#                 None,
#                 label=f"* {param['name']}: {param['value']}",
#                 size=13,
#                 color="darkgreen")

#             original_value = static_text_control(
#                 None,
#                 label=f"(default: {param_change.original_value})",
#                 color='darkgray',
#                 size=12)
#             hbox.addWidget(lbl)
#             hbox.addWidget(original_value)
#             self.layout.addLayout(hbox)

#     @property
#     def is_empty(self) -> bool:
#         """Boolean indicating whether there are any changes"""
#         return not bool(self.changes)

#     def update_changes(self, json_file: str):
#         """Update the changed items"""
#         self.clear()
#         self.changes = changes_from_default(json_file)
#         self._init_changes()

#     def show(self):
#         """Show the changes"""
#         show_all(self.layout)

#     def hide(self):
#         """Hide the changes"""
#         hide_all(self.layout)

#     def clear(self):
#         """Clear items"""
#         clear_layout(self.layout)

# class ParamsChanges(QWidget):
#     """Displays customizations from the default parameters in a scroll area
#     with buttons to show / hide the list of items."""

#     def __init__(self, json_file: str):
#         super().__init__()
#         self.change_items = ChangeItems(json_file)
#         self.collapsed = self.change_items.is_empty

#         self.show_label = '[+]'
#         self.hide_label = '[-]'

#         self.layout = QVBoxLayout()

#         self.changes_area = QScrollArea()
#         self.changes_area.setVerticalScrollBarPolicy(
#             Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#         self.changes_area.setHorizontalScrollBarPolicy(
#             Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#         self.changes_area.setWidgetResizable(True)
#         self.changes_area.setWidget(self.change_items)
#         self.changes_area.setVisible(not self.collapsed)

#         control_box = QHBoxLayout()
#         control_box.addWidget(
#             static_text_control(None, label='Changed Parameters:'))
#         self.toggle_button = QPushButton(
#             self.show_label if self.collapsed else self.hide_label)
#         self.toggle_button.setFlat(True)
#         self.toggle_button.setFixedWidth(40)
#         self.toggle_button.clicked.connect(self.toggle)
#         control_box.addWidget(self.toggle_button)

#         self.layout.addLayout(control_box)
#         self.layout.addWidget(self.changes_area)
#         self.setLayout(self.layout)

#     def update_changes(self, json_file: str):
#         """Update the changed items"""
#         self.change_items.update_changes(json_file)
#         self.changes_area.repaint()

#     def toggle(self):
#         """Toggle visibility of items"""
#         if self.collapsed:
#             self.show()
#         else:
#             self.collapse()
#         self.collapsed = not self.collapsed

#     def show(self):
#         """Show the changes"""
#         self.changes_area.setVisible(True)
#         self.toggle_button.setText(self.hide_label)

#     def collapse(self):
#         """Hide the changes"""
#         self.changes_area.setVisible(False)
#         self.toggle_button.setText(self.show_label)


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
        # self.changes = ParamsChanges(self.json_file)
        self.flash_msg = ''
        self.initUI()

    def initUI(self):
        """Initialize the UI"""
        vbox = QVBoxLayout()

        header_box = QHBoxLayout()
        header_box.addSpacing(5)
        self.edit_msg = static_text_control(self,
                                            label='Simulation Configuration',
                                            size=14,
                                            color='dimgray')
        header_box.addWidget(self.edit_msg)
        vbox.addLayout(header_box)

        # self.changes_panel = QHBoxLayout()
        # self.changes_panel.addWidget(self.changes)
        # vbox.addLayout(self.changes_panel)

        self.form_panel = QScrollArea()
        self.form_panel.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.form_panel.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.form_panel.setWidgetResizable(True)
        self.form_panel.setFixedWidth(self.size[0])
        self.form_panel.setWidget(self.form)

        vbox.addWidget(self.form_panel)
        vbox.addSpacing(5)

        # Controls
        control_box = QHBoxLayout()
        control_box.addStretch()
        run_button = QPushButton('Run')
        run_button.setFixedWidth(80)
        run_button.clicked.connect(self.on_run)
        control_box.addWidget(run_button)

        vbox.addLayout(control_box)

        self.save_msg = static_text_control(self,
                                            label='',
                                            size=12,
                                            color='darkgray')
        vbox.addWidget(self.save_msg)
        self.setLayout(vbox)
        self.setFixedHeight(self.size[1])
        self.setWindowTitle(self.title)
        self.show()

    def on_run(self):
        """Event handler for running the simulation."""
        if self.form.save():
            self.save_msg.setText(f'Last saved: {datetime.now()}')
            self.save_msg.repaint()

            # self.changes.update_changes(self.json_file)
            # self.changes.repaint()


def init(title='BCI Simulator', size=(750, 800)) -> str:
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
