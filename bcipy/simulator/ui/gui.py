"""Graphical User Interface for configuring the Simulator."""

# pylint: disable=E0611
import argparse
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from bcipy.core.parameters import Parameters
from bcipy.gui.file_dialog import FileDialog
from bcipy.gui.main import static_text_control
from bcipy.helpers.acquisition import active_content_types
from bcipy.io.load import load_json_parameters
from bcipy.preferences import preferences
from bcipy.simulator.data.sampler import TargetNontargetSampler
from bcipy.simulator.data.sampler.base_sampler import Sampler
from bcipy.simulator.task.copy_phrase import SimulatorCopyPhraseTask
from bcipy.simulator.task.task_factory import TaskFactory
from bcipy.simulator.ui.cli import excluded
from bcipy.simulator.ui.gui_utils import sampler_options
from bcipy.simulator.ui.obj_args_widget import ObjectArgInputs
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
                 parent: Optional[QWidget] = None,
                 parent_directory: Optional[str] = None,
                 selected_subdirectories: Optional[List[Path]] = None):
        super().__init__(parent=parent)
        self.parent_directory = parent_directory
        self.paths = selected_subdirectories or []

        self.label = "Selected Directories"
        self.tree = self.make_tree()
        self.box_layout = QVBoxLayout()
        self.box_layout.addWidget(
            LabeledWidget(self.label, self.tree, label_size=12))
        self.box_layout.addWidget(self.tree)
        self.setLayout(self.box_layout)

    def update_tree(self, parent_directory: Optional[str],
                    selected_subdirectories: Optional[List[Path]]) -> None:
        """Update the widget."""
        self.parent_directory = parent_directory
        self.paths = selected_subdirectories or []

        clear_layout(self.box_layout)
        if self.parent_directory:
            self.tree = self.make_tree()
            self.box_layout.addWidget(
                LabeledWidget(self.label, self.tree, label_size=12))

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
                 parent: Optional[QWidget] = None,
                 value: Optional[str] = None,
                 file_selector: str = "*",
                 prompt: str = "Select a file",
                 change_event: Optional[Callable] = None):

        super().__init__(parent=parent)
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
                 parent: Optional[QWidget] = None,
                 value: Optional[str] = None,
                 prompt: str = "Select a directory",
                 change_event: Optional[Callable] = None):
        super().__init__(parent=parent,
                         value=value,
                         prompt=prompt,
                         change_event=change_event)

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

    def __init__(self,
                 parent: Optional[QWidget] = None,
                 change_event: Optional[Callable] = None,
                 **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.change_event = change_event
        self.box_layout = QVBoxLayout()

        self.nested_filter_control = QCheckBox("Include nested directories")
        self.nested_filter_control.setChecked(True)
        self.nested_filter_control.checkStateChanged.connect(self.change)

        # self.name_filter_control_label = QLabel("Name contains")
        self.name_filter_control = QLineEdit("")
        self.name_filter_control.textChanged.connect(self.change)
        self.box_layout.addWidget(self.nested_filter_control)

        form = QFormLayout()
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.addRow("Name contains", self.name_filter_control)

        self.box_layout.addLayout(form)
        self.setLayout(self.box_layout)

    def change(self):
        """Called when a filter field is modified."""
        if self.change_event:
            self.change_event()


class DataDirectorySelect(QWidget):
    """Widget that facilitates the selection of data directories.
    Includes an input for a parent directory and optional filters.
    """

    def __init__(self,
                 parent: Optional[QWidget] = None,
                 parent_directory: Optional[str] = None,
                 change_event: Optional[Callable] = None):
        super().__init__(parent=parent)
        vbox = QVBoxLayout()
        self.change_event = change_event
        self.filters_enabled = False
        self.parent_directory_control = ChooseDirectoryInput(
            change_event=self.update_directory_tree,
            prompt="Select a parent data directory",
            value=parent_directory)

        self.directory_filter_control = DirectoryFilters(
            change_event=self.update_directory_tree)
        self.directory_tree = DirectoryTree()

        parent_layout = self.parent_directory_control.layout()
        filter_layout = self.directory_filter_control.layout()
        if parent_layout:
            parent_layout.setContentsMargins(0, 0, 0, 0)
        if filter_layout:
            filter_layout.setContentsMargins(0, 0, 0, 0)
        self.directory_filter_control.setEnabled(self.filters_enabled)
        self.directory_tree.setEnabled(self.filters_enabled)

        form = QFormLayout()
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft
                              | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Data Parent Directory", self.parent_directory_control)
        vbox.addLayout(form)
        vbox.addWidget(self.directory_filter_control)
        vbox.addWidget(self.directory_tree)
        self.setLayout(vbox)

    def parent_directory(self) -> Optional[str]:
        """Parent directory"""
        parent = self.parent_directory_control.value()
        if parent:
            return parent.strip()
        return None

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

        if self.parent_directory():
            if not self.filters_enabled:
                self.filters_enabled = True
                self.directory_filter_control.setEnabled(True)
                self.directory_tree.setEnabled(True)
            self.directory_tree.update_tree(self.parent_directory(),
                                            self.data_directories())
        else:
            self.filters_enabled = False
            self.directory_filter_control.setEnabled(False)
            self.directory_tree.setEnabled(False)
        # notify any listeners of the change
        if self.change_event:
            self.change_event()


class ModelFileInput(ChooseFileInput):
    """Prompts for a parameters file."""

    def __init__(self,
                 parent: Optional[QWidget] = None,
                 value: Optional[str] = None,
                 file_selector: str = "*.pkl",
                 prompt: str = "Select a model",
                 change_event: Optional[Callable] = None):
        super().__init__(parent=parent,
                         value=value,
                         file_selector=file_selector,
                         prompt=prompt,
                         change_event=change_event)


class ModelInputs(QWidget):
    """Provide a model path input for each configured content type."""

    def __init__(self,
                 parent: Optional[QWidget] = None,
                 content_types: Optional[List[str]] = None,
                 change_event: Optional[Callable] = None):
        super().__init__(parent=parent)
        self.change_event = change_event
        self.form_layout = QFormLayout()
        self.form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft
                                          | Qt.AlignmentFlag.AlignVCenter)
        self.form_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.content_types = content_types or ['EEG']
        self.controls = self._create_inputs()
        self._add_controls()
        self.setLayout(self.form_layout)

    def value(self) -> List[str]:
        """Model paths"""
        return [control[1].value() for control in self.controls]

    def set_content_types(self, value: List[str]):
        """Set the content_content types and re-generate inputs."""
        self.content_types = value
        self.controls = self._create_inputs(initialized=True)
        self._add_controls()

    def _create_inputs(self,
                       initialized: bool = False) -> List[Tuple[str, QWidget]]:
        """Create a path input to a model for each content type"""
        return [(self._label(content_type),
                 ModelFileInput(
                     parent=self,
                     change_event=self.change,
                     value=self._path(content_type) if initialized else None))
                for content_type in self.content_types]

    def _path(self, content_type: str) -> Optional[str]:
        """Get the input path for the provided content type"""
        matches = [
            control for control in self.controls
            if control[0] == self._label(content_type)
        ]
        if matches:
            return matches[0][1].value()
        return None

    def _label(self, content_type: str) -> str:
        return f"{content_type} Model"

    def _add_controls(self) -> None:
        """Add each model input to the layout."""
        clear_layout(self.form_layout)
        for (label, control) in self.controls:
            self.form_layout.addRow(label, control)

    def change(self):
        """Called when a model path changes"""
        if self.change_event:
            self.change_event()


class SelectionList(QWidget):
    """Widget for selecting a sampler."""

    def __init__(self,
                 parent: Optional[QWidget] = None,
                 change_event: Optional[Callable] = None,
                 items: Optional[List[str]] = None):
        super().__init__(parent=parent)
        self.change_event = change_event
        self.form_layout = QFormLayout()
        self.control = QComboBox()
        if items:
            self.control.addItems(items)
        self.control.currentTextChanged.connect(self.change)

        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.control)
        self.setLayout(hbox)

    def value(self) -> str:
        """Selected value"""
        return self.control.currentText()

    def change(self):
        """Called when a model path changes"""
        if self.change_event:
            self.change_event()


class LabeledWidget(QWidget):
    """Renders a widget with a label above it."""

    def __init__(self,
                 label: str,
                 widget: QWidget,
                 label_size: int = 14,
                 parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        vbox = QVBoxLayout()
        lbl = QLabel()
        lbl.setText(label)
        if label_size:
            lbl.setFont(QFont(None, label_size))

        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(lbl)
        vbox.addWidget(widget)
        self.setLayout(vbox)


def sim_runs_control(value: int = 1) -> QWidget:
    """Create a widget for entering simulation runs."""
    spin_box = QSpinBox()
    spin_box.setMinimum(1)
    spin_box.setMaximum(1000)
    # disable scroll wheel
    spin_box.wheelEvent = lambda event: None  # type: ignore
    spin_box.setValue(value)
    return spin_box


class SimConfigForm(QWidget):
    """The InputForm class is a QWidget that creates controls/inputs for each simulation parameter

  Parameters:
  -----------
    json_file - path of parameters file to be edited.
    width - optional; used to set the width of the form controls.
  """

    def __init__(self,
                 width: int = 400,
                 change_event: Optional[Callable] = None):
        super().__init__()

        vbox = QVBoxLayout()
        self.setFixedWidth(width)

        self.change_event = change_event
        self.parameters_path: Optional[str] = None
        self.model_paths: List[str] = []
        self.data_paths: Optional[List[Path]] = []

        self.sampler: Optional[Sampler] = TargetNontargetSampler
        self.sampler_options = sampler_options(default=self.sampler)
        self.sampler_args = '{}'

        self.runs_control = sim_runs_control()
        self.output_control = ChooseDirectoryInput(
            value=DEFAULT_SAVE_LOCATION, change_event=self.update_data_paths)

        self.parameter_control = ParameterFileInput(
            change_event=self.update_parameters)
        self.directory_control = DataDirectorySelect(
            change_event=self.update_data_paths)
        self.model_input_control = ModelInputs(change_event=self.update_models)
        self.sampler_input_control = SelectionList(
            change_event=self.update_sampler,
            items=list(self.sampler_options.keys()))
        self.sampler_args_control = ObjectArgInputs(
            change_event=self.update_sampler_args, object_type=self.sampler)

        form = QFormLayout()
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft
                              | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Simulation Runs:", self.runs_control)
        form.addRow("Output Directory:", self.output_control)
        form.addRow("Sim Parameters:", self.parameter_control)

        vbox.addLayout(form)
        vbox.addSpacing(2)
        vbox.addWidget(self.create_model_group())
        vbox.addSpacing(2)
        vbox.addWidget(self.create_sampler_group())
        vbox.addSpacing(2)
        vbox.addWidget(self.create_data_group())
        self.setLayout(vbox)
        self.show()

    def create_sampler_group(self) -> QWidget:
        """Create a group box for model inputs"""
        group = QGroupBox("Sampler")
        vbox = QVBoxLayout()
        vbox.addWidget(self.sampler_input_control)
        vbox.addWidget(self.sampler_args_control)
        group.setLayout(vbox)
        return group

    def create_data_group(self) -> QWidget:
        """Create a group box for data inputs."""
        group_box = QGroupBox("Data")
        vbox = QVBoxLayout()
        vbox.addWidget(self.directory_control)
        group_box.setLayout(vbox)
        return group_box

    def create_model_group(self) -> QWidget:
        """Create a group box for model inputs"""
        group = QGroupBox("Models")
        vbox = QVBoxLayout()
        vbox.addWidget(self.model_input_control)
        group.setLayout(vbox)
        return group

    def runs(self) -> int:
        """Number of runs"""
        return int(self.runs_control.text())

    def outdir(self) -> Optional[str]:
        """Output directory"""
        return self.output_control.value()

    def command_valid(self) -> bool:
        """Returns True if all necessary fields are input."""
        return bool(self.parameters_path and self.model_paths
                    and self.data_paths and self.sampler and self.sampler_args)

    def command(self) -> str:
        """Command equivalent to to the result of the interactive selection of
        simulator inputs."""
        if not self.command_valid():
            return ''

        args = []
        if self.outdir():
            args.append(f"-o '{self.outdir()}'")
        args.append(f"-p '{self.parameters_path}'")
        args.extend([f"-m '{path}'" for path in self.model_paths])
        if self.data_paths:
            args.extend([f"-d '{source}'" for source in self.data_paths])

        args.append(f"-n {self.runs()}")
        if self.sampler is not None:
            args.append(f"-s {self.sampler.__name__}")
        args.append(f"--sampler_args='{self.sampler_args}'")
        args.append("-v")

        return f"bcipy-sim {' '.join(args)}"

    def update_parameters(self) -> None:
        """When simulation parameters are updated include an input for each model."""
        self.parameters_path = self.parameter_control.value()
        params = self.parameter_control.parameters
        if params:
            content_types = active_content_types(params.get('acq_mode', 'EEG'))
        else:
            content_types = ['EEG']
        self.model_input_control.set_content_types(content_types)
        self.change()

    def update_models(self) -> None:
        """Update the model paths from the input controls"""
        self.model_paths = self.model_input_control.value()
        self.change()

    def update_data_paths(self) -> None:
        """Update the data paths from the directory inputs"""
        self.data_paths = self.directory_control.data_directories()
        self.change()

    def update_sampler(self) -> None:
        """Called when the sampler is updated. Updates the sampler args input"""
        selected_sampler = self.sampler_options.get(
            self.sampler_input_control.value(), None)
        self.sampler = selected_sampler
        self.sampler_args_control.set_object_type(selected_sampler)
        self.change()

    def update_sampler_args(self) -> None:
        """Update the sampler args"""
        if self.sampler_args_control.required_inputs_provided():
            self.sampler_args = self.sampler_args_control.value()
        else:
            self.sampler_args = ''
        self.change()

    def change(self):
        """Announce change to any registered change events."""
        if self.change_event:
            self.change_event()


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
        self.w_size = size

        self.command = ''

        self.form = SimConfigForm(width=self.w_size[0],
                                  change_event=self.on_params_change)
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
        vbox.addWidget(self.form)
        vbox.addSpacing(4)

        # Controls
        control_box = QHBoxLayout()
        control_box.addStretch()
        self.run_button = QPushButton('Run')
        self.run_button.setFixedWidth(80)
        self.run_button.clicked.connect(self.on_run)
        self.run_button.setEnabled(False)
        control_box.addWidget(self.run_button)

        vbox.addLayout(control_box)

        self.command_msg = static_text_control(self,
                                               label='',
                                               size=12,
                                               color='darkgray')
        vbox.addWidget(self.command_msg)
        self.setLayout(vbox)
        self.setFixedHeight(self.w_size[1])
        self.setWindowTitle(self.title)
        self.show()

    def cli_output(self) -> str:
        """Returns the command with arguments."""
        if self.form.command_valid():
            return self.form.command()
        return ''

    def on_run(self):
        """Event handler for running the simulation."""

        if self.form.command_valid():
            self.command_msg.setText(self.cli_output())
            self.command_msg.repaint()
            self.close()

    def on_params_change(self) -> None:
        """Event performed when form parameters change."""
        if self.form.command_valid():
            self.run_button.setEnabled(True)
        else:
            self.run_button.setEnabled(False)


def init(title='BCI Simulator', size=(900, 600)) -> str:
    """Set up the GUI components and start the main loop."""
    app = QApplication(sys.argv)
    panel = MainPanel(title, size)
    app.exec()
    command = panel.cli_output()
    app.quit()
    if command:
        print(command, file=sys.stdout)
        subprocess.run(command, shell=True)
    return command


def configure(
    title='BCI Simulator',
    size=(600, 900)
) -> Tuple[int, Optional[str], Optional[TaskFactory]]:
    """Main function"""
    app = QApplication(sys.argv)
    panel = MainPanel(title, size)
    app.exec()
    command = panel.cli_output()

    runs = panel.form.runs()
    outdir = panel.form.outdir()
    params = panel.form.parameters_path
    data_paths = panel.form.data_paths
    model_paths = panel.form.model_paths
    sampler = panel.form.sampler
    sampler_args = panel.form.sampler_args

    app.quit()

    factory = None
    if command:
        print(command, file=sys.stdout)
        parameters = load_json_parameters(params, value_cast=True)
        factory = TaskFactory(parameters=parameters,
                              source_dirs=data_paths,
                              signal_model_paths=model_paths,
                              sampling_strategy=sampler,
                              sampler_args=json.loads(sampler_args),
                              task=SimulatorCopyPhraseTask)
    return (runs, outdir, factory)


def run():
    """Process command line arguments and initialize the GUI."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--width', default=600, type=int)
    parser.add_argument('--height', default=900, type=int)

    args = parser.parse_args()
    init(size=(args.width, args.height))


if __name__ == '__main__':
    run()
