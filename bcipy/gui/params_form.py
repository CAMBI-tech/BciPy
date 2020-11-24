"""GUI form for editing a Parameters file."""
# pylint: disable=E0611
import logging
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QScrollArea, QVBoxLayout, QWidget)

from bcipy.helpers.parameters import Parameters

# Holds the label, help, and input controls for a given parameter.
FormInput = namedtuple('FormInput', ['control', 'label', 'help'])
Parameter = namedtuple('Parameter', [
    'value', 'section', 'readableName', 'helpTip', 'recommended_values', 'type'
])


def font(size: int = 14, font_family: str = 'Helvetica') -> QFont:
    """Create a Font object with the given parameters."""
    return QFont(font_family, size, QFont.Normal)


def static_text_control(parent,
                        label: str,
                        color: str = 'black',
                        size: int = 14,
                        font_family: str = 'Helvetica') -> QLabel:
    """Creates a static text control with the given font parameters. Useful for
    creating labels and help components."""
    static_text = QLabel(parent)
    static_text.setWordWrap(True)
    static_text.setText(label)
    static_text.setStyleSheet(f'color: {color};')
    static_text.setFont(font(size, font_family))
    return static_text


def get_directory():
    """Prompts the user with a FileDialog. Returns the directory name
        if one was selected."""
    return QFileDialog.getExistingDirectory(caption='Select a path')


def get_file():
    """Prompts the user with a FileDialog. Returns the file name
        if one was selected."""
    name, _ = QFileDialog.getOpenFileName(caption='Select a file',
                                          filter='All Files (*)')
    return name


def button_handler(get_value_fn, update_fn):
    """Returns a callback function that updates the parameter for the
provided control.
"""

    def handler():
        value = get_value_fn()
        if value:
            update_fn(value)

    return handler


def init_combobox(control: QComboBox, options: List[str], selected_value: str):
    """Initialize the given combobox."""
    if selected_value not in options:
        options = [selected_value] + options
    control.clear()
    control.addItems(options)
    control.setCurrentIndex(options.index(selected_value))
    return control


def input_label(param: Parameter,
                help_font_size: int = 12,
                help_color: str = 'darkgray') -> Tuple[QLabel, QLabel]:
    """Returns a label control and maybe a help Control if the help
    text is different than the label text."""
    label = static_text_control(None, label=param.readableName)
    help_tip = None
    if param.readableName != param.helpTip:
        help_tip = static_text_control(None,
                                       label=param.helpTip,
                                       size=help_font_size,
                                       color=help_color)
    return (label, help_tip)


# Input Types
def bool_input(param: Parameter) -> FormInput:
    """Creates a checkbox FormInput"""
    ctl = QCheckBox(param.readableName)
    ctl.setChecked(param.value == 'true')
    ctl.setFont(font())
    return FormInput(ctl, label=None, help=None)


def selection_input(param: Parameter) -> FormInput:
    """Creates a selection pulldown FormInput."""
    ctl = init_combobox(QComboBox(), param.recommended_values, param.value)
    ctl.setEditable(True)
    label, help_tip = input_label(param)
    return FormInput(ctl, label, help_tip)


def text_input(param: Parameter) -> FormInput:
    """Creates a text field FormInput."""
    ctl = QLineEdit(param.value)
    label, help_tip = input_label(param)
    return FormInput(ctl, label, help_tip)


def file_input(param: Parameter) -> FormInput:
    """File Input.

    Creates a text field or selection pulldown FormInput with a button
    to browse for a file."""
    # Creates a combobox instead of text field if the parameter has
    # recommended values.
    if isinstance(param.recommended_values, list):
        options = param.recommended_values
        ctl = init_combobox(QComboBox(), options, param.value)
        ctl.setEditable(True)

        def update_fn(name):
            return init_combobox(ctl, options, name)
    else:
        ctl = QLineEdit(param.value)

        def update_fn(name):
            return ctl.setText(name)

    is_directory = param.type == 'directorypath'
    get_value_fn = get_directory if is_directory else get_file
    btn = QPushButton('...')
    btn.setFixedWidth(40)
    btn.clicked.connect(button_handler(get_value_fn, update_fn))

    ctl_array = [ctl, btn]

    label, help_tip = input_label(param)
    return FormInput(ctl_array, label, help_tip)


class ParamsForm(QWidget):
    """The Form class is a wx.Panel that creates controls/inputs for each
  parameter in the provided json file.

  Parameters:
  -----------
    json_file - path of parameters file to be edited.
    load_file - optional path of parameters file to load;
        parameters from this file will be copied over to the json_file.
    control_width - optional; used to set the size of the form controls.
    control_height - optional; used to set the size of the form controls.
  """

    def __init__(self, json_file: str, load_file: str = None,
                 width: int = 400):
        super().__init__()
        self.json_file = json_file
        self.load_file = json_file if not load_file else load_file
        self.width = width
        self.help_font_size = 12
        self.help_color = 'darkgray'

        self.params = Parameters(source=self.load_file, cast_values=False)

        self.controls = {}
        self.create_controls()
        self.do_layout()

    def create_controls(self):
        """Create controls (inputs, labels, etc) for each item in the
    parameters file.

    TODO: include a search box for finding an input
    """
        for key, param in self.params.entries():
            param = Parameter(**param)
            if param.type == 'bool':
                form_input = bool_input(param)
            elif 'path' in param.type:
                form_input = file_input(param)
            elif isinstance(param.recommended_values, list):
                form_input = selection_input(param)
            else:
                form_input = text_input(param)

            self.add_input(key, form_input)

    def add_input(self, key: str, form_input: FormInput) -> None:
        """Adds the controls for the given input to the controls structure."""
        if form_input.label:
            self.controls[f'{key}_label'] = form_input.label
        if form_input.help:
            self.controls[f'{key}_help'] = form_input.help
        self.controls[key] = form_input.control

    def do_layout(self):
        """Layout the form controls."""
        vbox = QVBoxLayout()

        # Add the controls to the grid:
        for control_name, control in self.controls.items():
            if isinstance(control, list):
                hbox = QHBoxLayout()
                hbox.addWidget(control[0])
                hbox.addWidget(control[1])
                vbox.addLayout(hbox)
            else:
                # control.setFixedWidth(self.control_width)
                vbox.addWidget(control)

            # Add space after the input
            if control_name in self.params:
                vbox.addSpacing(10)

        self.setLayout(vbox)
        self.setFixedWidth(self.width)
        self.show()

    def save(self) -> bool:
        """Save changes"""
        self.update_parameters()
        path = Path(self.json_file)
        self.params.save(path.parent, path.name)
        return True

    def update_parameters(self):
        """Update the parameters from the input values."""

        for key, control in self.controls.items():
            # Only consider inputs, not label and help controls.
            if key in self.params:
                if isinstance(control, list):
                    # File/directory input
                    control = control[0]

                param = self.params[key]
                if param['type'] == 'bool':
                    # Checkbox
                    value = 'true' if control.isChecked() else 'false'
                elif isinstance(param['recommended_values'], list):
                    # combobox
                    value = control.currentText()
                else:
                    value = control.text()

                if value != param['value']:
                    print(key, ": ", value)

                self.params[key]['value'] = value


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
        self.flash_msg = ''
        self.initUI()

    def initUI(self):
        """Initialize the UI"""
        vbox = QVBoxLayout()
        # vbox.addStretch(1)
        vbox.addWidget(static_text_control(self, label=self.title, size=20))

        vbox.addWidget(
            static_text_control(self,
                                label=f'Editing: {self.json_file}',
                                size=14))

        load_button = QPushButton('Load Values')
        load_button.clicked.connect(self.on_load)
        vbox.addWidget(load_button)

        self.load_msg = static_text_control(self,
                                            label='',
                                            size=14,
                                            color='green')
        vbox.addWidget(self.load_msg)
        vbox.addSpacing(10)

        self.form_panel = QScrollArea()
        self.form_panel.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.form_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.form_panel.setWidgetResizable(True)
        self.form_panel.setFixedWidth(self.size[0])
        self.form_panel.setWidget(self.form)

        vbox.addWidget(self.form_panel)
        vbox.addSpacing(5)

        save_button = QPushButton('Save')
        save_button.clicked.connect(self.on_save)
        vbox.addWidget(save_button)

        self.save_msg = static_text_control(self,
                                            label='',
                                            size=12,
                                            color='darkgray')
        vbox.addWidget(self.save_msg)
        self.setLayout(vbox)
        self.setFixedHeight(self.size[1])
        self.setWindowTitle(self.title)
        self.show()

    def on_load(self):
        """Event handler to load the form data from a different json file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption='Open parameters file',
            directory='bcipy/parameters',
            filter='JSON files (*.json)')
        if file_name:
            self.load_msg.setText(
                f'Loaded from: {file_name}. Click the Save button to persist these changes.'
            )
            self.load_msg.repaint()
            self.form = ParamsForm(json_file=self.json_file,
                                   load_file=file_name,
                                   width=self.size[0])
            self.form_panel.setWidget(self.form)

    def on_save(self):
        """Event handler for saving form data"""
        if self.form.save():
            self.save_msg.setText(f'Last saved: {datetime.now()}')
            self.save_msg.repaint()


def main(json_file, title='BCI Parameters', size=(450, 550)):
    """Set up the GUI components and start the main loop."""
    app = QApplication(sys.argv)
    _panel = MainPanel(json_file, title, size)
    sys.exit(app.exec_())


if __name__ == '__main__':

    import argparse
    from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH

    parser = argparse.ArgumentParser()

    # Command line utility for adding arguments/ paths via command line
    parser.add_argument('-p',
                        '--parameters',
                        default=DEFAULT_PARAMETERS_PATH,
                        help='Path to parameters.json configuration file.')
    args = parser.parse_args()
    main(args.parameters)
