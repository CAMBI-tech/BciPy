"""GUI form for editing a Parameters file."""
# pylint: disable=E0611
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout,
                             QPushButton, QScrollArea, QVBoxLayout, QWidget)
from bcipy.helpers.parameters import Parameters, changes_from_default

from bcipy.gui.main import (
    BoolInput,
    DirectoryInput,
    FileInput,
    FloatInput,
    FormInput,
    IntegerInput,
    SearchInput,
    SelectionInput,
    static_text_control,
    TextInput,
)


class ParamsForm(QWidget):
    """The ParamsForm class is a QWidget that creates controls/inputs for each parameter in the
    provided json file.

  Parameters:
  -----------
    json_file - path of parameters file to be edited.
    width - optional; used to set the width of the form controls.
  """

    def __init__(self, json_file: str, width: int = 400):
        super().__init__()

        self.help_size = 12
        self.help_color = 'darkgray'
        self.search_text = ''

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setFixedWidth(width)

        self.json_file = json_file
        self.params = Parameters(source=self.json_file, cast_values=False)
        self.controls = self.create_controls(self.params)
        self.add_controls()

        self.show()

    def add_controls(self):
        """Add controls to layout"""
        if self.controls:
            for _param_name, form_input in self.controls.items():
                self.layout.addWidget(form_input)

    def create_controls(self, params: Parameters) -> Dict[str, FormInput]:
        """Create controls (inputs, labels, etc) for each item in the
    parameters file.
    """
        controls = {}
        for key, param in params.entries():
            controls[key] = self.parameter_input(param)
        return controls

    def update(self, json_file: str):
        """Update the form using a new json_file."""
        clear_layout(self.layout)
        self.json_file = json_file
        self.params = Parameters(source=self.json_file, cast_values=False)
        self.controls = self.create_controls(self.params)
        self.add_controls()
        self.search(self.search_text)

    def parameter_input(self, param: Dict[str, str]) -> FormInput:
        """Construct a FormInput for the given parameter based on its python type and other
        attributes."""

        type_inputs = {
            'int': IntegerInput,
            'float': FloatInput,
            'bool': BoolInput,
            'filepath': FileInput,
            'directorypath': DirectoryInput
        }
        has_options = isinstance(param['recommended_values'], list)
        form_input = type_inputs.get(
            param['type'], SelectionInput if has_options else TextInput)
        return form_input(label=param['readableName'],
                          value=param['value'],
                          help_tip=param['helpTip'],
                          options=param['recommended_values'],
                          help_size=self.help_size,
                          help_color=self.help_color)

    def search(self, text: str) -> None:
        """Search for an input. Hides inputs which do not match.
        Searches labels, hints, and values.

        Parameter:
        ----------
          text - text used to search; if empty all inputs are displayed.
         """
        self.search_text = text
        for form_input in self.controls.values():
            if text == '' or form_input.matches(text):
                form_input.show()
            else:
                form_input.hide()

    def save(self) -> bool:
        """Save changes"""
        self.update_parameters()
        path = Path(self.json_file)
        self.params.save(path.parent, path.name)
        return True

    def save_as(self, filename: str) -> bool:
        """Save parameters to another file."""
        assert filename, "Filename is required"
        self.update_parameters()

        self.json_file = filename
        path = Path(self.json_file)
        self.params.save(path.parent, path.name)
        return True

    def update_parameters(self):
        """Update the parameters from the input values."""

        for param_name, form_input in self.controls.items():
            param = self.params[param_name]
            value = form_input.value()
            if value != param['value']:
                self.params[param_name]['value'] = value


def clear_layout(layout):
    """Clear all items from a layout.
    https://stackoverflow.com/questions/9374063/remove-all-items-from-a-layout
    """
    while layout.count():
        child = layout.takeAt(0)
        if child.widget() is not None:
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


class ChangeItems(QWidget):
    """Line items showing parameter changes."""

    def __init__(self, json_file: str):
        super().__init__()
        self.changes = changes_from_default(json_file)
        self.layout = QVBoxLayout()
        self._init_changes()
        self.setLayout(self.layout)

    def _init_changes(self):
        """Initialize the line items for changes."""
        if not self.changes:
            self.layout.addWidget(
                static_text_control(None,
                                    label="None",
                                    size=12,
                                    color='darkgray'))

        for _key, param_change in self.changes.items():
            param = param_change.parameter
            hbox = QHBoxLayout()

            lbl = static_text_control(
                None,
                label=f"* {param['readableName']}: {param['value']}",
                size=13,
                color="darkgreen")

            original_value = static_text_control(
                None,
                label=f"(default: {param_change.original_value})",
                color='darkgray',
                size=12)
            hbox.addWidget(lbl)
            hbox.addWidget(original_value)
            self.layout.addLayout(hbox)

    @property
    def is_empty(self) -> bool:
        """Boolean indicating whether there are any changes"""
        return not bool(self.changes)

    def update_changes(self, json_file: str):
        """Update the changed items"""
        self.clear()
        self.changes = changes_from_default(json_file)
        self._init_changes()

    def show(self):
        """Show the changes"""
        show_all(self.layout)

    def hide(self):
        """Hide the changes"""
        hide_all(self.layout)

    def clear(self):
        """Clear items"""
        clear_layout(self.layout)


class ParamsChanges(QWidget):
    """Displays customizations from the default parameters in a scroll area
    with buttons to show / hide the list of items."""

    def __init__(self, json_file: str):
        super().__init__()
        self.change_items = ChangeItems(json_file)
        self.collapsed = self.change_items.is_empty

        self.show_label = '[+]'
        self.hide_label = '[-]'

        self.layout = QVBoxLayout()

        self.changes_area = QScrollArea()
        self.changes_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.changes_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.changes_area.setWidgetResizable(True)
        self.changes_area.setWidget(self.change_items)
        self.changes_area.setVisible(not self.collapsed)

        control_box = QHBoxLayout()
        control_box.addWidget(
            static_text_control(None, label='Changed Parameters:'))
        self.toggle_button = QPushButton(
            self.show_label if self.collapsed else self.hide_label)
        self.toggle_button.setFlat(True)
        self.toggle_button.setFixedWidth(40)
        self.toggle_button.clicked.connect(self.toggle)
        control_box.addWidget(self.toggle_button)

        self.layout.addLayout(control_box)
        self.layout.addWidget(self.changes_area)
        self.setLayout(self.layout)

    def update_changes(self, json_file: str):
        """Update the changed items"""
        self.change_items.update_changes(json_file)
        self.changes_area.repaint()

    def toggle(self):
        """Toggle visibility of items"""
        if self.collapsed:
            self.show()
        else:
            self.collapse()
        self.collapsed = not self.collapsed

    def show(self):
        """Show the changes"""
        self.changes_area.setVisible(True)
        self.toggle_button.setText(self.hide_label)

    def collapse(self):
        """Hide the changes"""
        self.changes_area.setVisible(False)
        self.toggle_button.setText(self.show_label)


class MainPanel(QWidget):
    """Main GUI window.
    Parameters:
    -----------
      json_file: path to the parameters file.
      title: window title
      size: window size; (width, height)
    """

    def __init__(self, json_file: str, title: str, size: Tuple[int, int]):
        super().__init__()
        self.json_file = json_file
        self.title = title
        self.size = size

        self.form = ParamsForm(json_file, width=self.size[0])
        self.changes = ParamsChanges(self.json_file)
        self.flash_msg = ''
        self.initUI()

    def initUI(self):
        """Initialize the UI"""
        vbox = QVBoxLayout()

        header_box = QHBoxLayout()
        header_box.addSpacing(5)
        self.edit_msg = static_text_control(self,
                                            label=f'Editing: {self.json_file}',
                                            size=14,
                                            color='dimgray')
        header_box.addWidget(self.edit_msg)
        vbox.addLayout(header_box)

        vbox.addWidget(SearchInput(self.search_form))

        self.changes_panel = QHBoxLayout()
        self.changes_panel.addWidget(self.changes)
        vbox.addLayout(self.changes_panel)

        self.form_panel = QScrollArea()
        self.form_panel.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.form_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.form_panel.setWidgetResizable(True)
        self.form_panel.setFixedWidth(self.size[0])
        self.form_panel.setWidget(self.form)

        vbox.addWidget(self.form_panel)
        vbox.addSpacing(5)

        # Controls
        control_box = QHBoxLayout()
        control_box.addStretch()
        save_button = QPushButton('Save')
        save_button.setFixedWidth(80)
        save_button.clicked.connect(self.on_save)
        control_box.addWidget(save_button)

        save_as_button = QPushButton('Save As')
        save_as_button.setFixedWidth(80)
        save_as_button.clicked.connect(self.on_save_as)
        control_box.addWidget(save_as_button)

        load_button = QPushButton('Open')
        load_button.setFixedWidth(80)
        load_button.clicked.connect(self.on_load)
        control_box.addWidget(load_button)
        control_box.addStretch()

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

    def search_form(self, text):
        """Search for a parameter."""
        self.form.search(text)

    def on_load(self):
        """Event handler to load the form data from a different json file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption='Open parameters file',
            directory='bcipy/parameters',
            filter='JSON files (*.json)')
        if file_name:
            self.json_file = file_name
            self.edit_msg.setText(f'Editing: {self.json_file}')
            self.form.update(self.json_file)
            self.changes.update_changes(self.json_file)
            self.repaint()

    def on_save(self):
        """Event handler for saving form data"""
        if self.form.save():
            self.save_msg.setText(f'Last saved: {datetime.now()}')
            self.save_msg.repaint()

            self.changes.update_changes(self.json_file)
            self.changes.repaint()

    def on_save_as(self):
        """Event handler for saving form data to another parameters file."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            caption='Save As',
            directory='bcipy/parameters/',
            filter='JSON files (*.json)',
            options=options)

        if filename and self.form.save_as(filename):
            self.json_file = filename
            self.edit_msg.setText(f'Editing: {self.json_file}')
            self.save_msg.setText(f'Last saved: {datetime.now()}')
            self.changes.update_changes(self.json_file)
            self.repaint()


def main(json_file, title='BCI Parameters', size=(450, 550)):
    """Set up the GUI components and start the main loop."""
    app = QApplication(sys.argv)
    _panel = MainPanel(json_file, title, size)
    sys.exit(app.exec_())


if __name__ == '__main__':

    import argparse
    from bcipy.config import DEFAULT_PARAMETERS_PATH

    parser = argparse.ArgumentParser()

    # Command line utility for adding arguments/ paths via command line
    parser.add_argument('-p',
                        '--parameters',
                        default=DEFAULT_PARAMETERS_PATH,
                        help='Path to parameters.json configuration file.')
    args = parser.parse_args()
    main(args.parameters)
