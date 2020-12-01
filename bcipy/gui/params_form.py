"""GUI form for editing a Parameters file."""
# pylint: disable=E0611
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

Parameter = namedtuple('Parameter', [
    'value', 'section', 'readableName', 'helpTip', 'recommended_values', 'type'
])

#--- Utility functions


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


class ComboBox(QComboBox):
    """ComboBox with the same interface as a QLineEdit for getting and setting values.

    Parameters:
    -----------
      options - list of items to appear in the lookup
      selected_value - selected item; if this is not one of the options and new option will be
        created for this value.
    """

    def __init__(self, options: List[str], selected_value: str, **kwargs):
        super(ComboBox, self).__init__(**kwargs)
        self.options = options
        self.addItems(self.options)
        self.setText(selected_value)
        self.setEditable(True)

    def setText(self, value: str):
        """Sets the current index to the given value. If the value is not in the list of
        options it will be added."""
        if value not in self.options:
            self.options = [value] + self.options
            self.clear()
            self.addItems(self.options)
        self.setCurrentIndex(self.options.index(value))

    def text(self):
        """Gets the currentText."""
        return self.currentText()


#--- Input widgets


class FormInput(QWidget):
    """A form element with a label, help tip, and input control. The default control is a text
    input. This object may be subclassed to specialize the input control or the arrangement of
    widgets. FormInputs have the ability to show and hide themselves.

    Parameters:
    ----------
      parameter - the option which will be configured using this input. Used to populate the label,
        help text, and relevant options.
      help_font_size - font size for the help text.
      help_color - color of the help text.
    """

    def __init__(self,
                 parameter: Parameter,
                 help_font_size: int = 12,
                 help_color: str = 'darkgray'):
        super(FormInput, self).__init__()
        self.parameter = parameter

        self.label = self.init_label()
        self.help_tip = self.init_help(help_font_size, help_color)
        self.control = self.init_control()

        self.init_layout()

    def init_label(self) -> QWidget:
        """Initize the label widget."""
        return static_text_control(None, label=self.parameter.readableName)

    def init_help(self, font_size: int, color: str) -> QWidget:
        """Initialize the help text widget."""
        if self.parameter.readableName == self.parameter.helpTip:
            return None
        return static_text_control(None,
                                   label=self.parameter.helpTip,
                                   size=font_size,
                                   color=color)

    def init_control(self) -> QWidget:
        """Initialize the form control widget."""
        # Default is a text input
        return QLineEdit(self.parameter.value)

    def init_layout(self):
        """Initialize the layout by adding the label, help, and control widgets."""
        self.vbox = QVBoxLayout()
        if self.label:
            self.vbox.addWidget(self.label)
        if self.help_tip:
            self.vbox.addWidget(self.help_tip)
        self.vbox.addWidget(self.control)
        self.setLayout(self.vbox)

    def value(self) -> str:
        """Returns the value associated with the form input."""
        if self.control:
            return self.control.text()
        return None

    def matches(self, term: str) -> bool:
        """Returns True if the input matches the given text, otherwise False."""
        text = term.lower()
        return (text in self.parameter.readableName.lower()) or (
            self.parameter.helpTip and text in self.parameter.helpTip.lower()
        ) or text in self.value().lower()

    def show(self):
        """Show this widget, and all child widgets."""
        for widget in self.widgets():
            if widget:
                widget.setVisible(True)

    def hide(self):
        """Hide this widget, and all child widgets."""
        for widget in self.widgets():
            if widget:
                widget.setVisible(False)

    def widgets(self) -> List[QWidget]:
        """Returns a list of self and child widgets. List may contain None values."""
        return [self.label, self.help_tip, self.control, self]


class BoolInput(FormInput):
    """Checkbox form input used for boolean configuration parameters. Overrides FormInput to provide
    a CheckBox control. Help text is not displayed for checkbox items."""

    def __init__(self, parameter: Parameter, **kwargs):
        super(BoolInput, self).__init__(parameter, **kwargs)

    def init_label(self) -> QWidget:
        """Override. Checkboxes do not have a separate label."""
        return None

    def init_help(self, font_size: int, color: str) -> QWidget:
        """Override. Checkboxes do not display help."""
        return None

    def init_control(self):
        """Override to create a checkbox."""
        ctl = QCheckBox(self.parameter.readableName)
        ctl.setChecked(self.parameter.value == 'true')
        ctl.setFont(font())
        return ctl

    def value(self) -> str:
        return 'true' if self.control.isChecked() else 'false'


class SelectionInput(FormInput):
    """Input to select from a list of options."""

    def __init__(self, parameter: Parameter, **kwargs):
        super(SelectionInput, self).__init__(parameter, **kwargs)

    def init_control(self) -> QWidget:
        """Override to create a Combobox."""
        return ComboBox(self.parameter.recommended_values,
                        self.parameter.value)


class TextInput(FormInput):
    """Text field input."""

    def __init__(self, parameter: Parameter, **kwargs):
        super(TextInput, self).__init__(parameter, **kwargs)


class FileInput(FormInput):
    """Input for selecting a file or directory."""

    def __init__(self, parameter: Parameter, **kwargs):
        super(FileInput, self).__init__(parameter, **kwargs)

    def init_control(self) -> QWidget:
        """Override to create either a selection list or text field depending
        on whether there are recommended values."""
        param = self.parameter
        if isinstance(self.parameter.recommended_values, list):
            return ComboBox(param.recommended_values, param.value)
        return QLineEdit(param.value)

    def init_button(self) -> QWidget:
        """Creates a Button to initiate the file/directory dialog."""
        btn = QPushButton('...')
        btn.setFixedWidth(40)
        btn.clicked.connect(self.prompt_path)
        return btn

    def prompt_path(self):
        """Prompts the user with a FileDialog. Updates the control's value with the file or
        directory if one was selected."""
        if self.parameter.type == 'directorypath':
            name = QFileDialog.getExistingDirectory(caption='Select a path')
        else:
            name, _ = QFileDialog.getOpenFileName(caption='Select a file',
                                                  filter='All Files (*)')

        if name:
            self.control.setText(name)

    def init_layout(self):
        """Overrides the layout to add the file dialog button."""
        self.button = self.init_button()
        self.vbox = QVBoxLayout()
        if self.label:
            self.vbox.addWidget(self.label)
        if self.help_tip:
            self.vbox.addWidget(self.help_tip)
        hbox = QHBoxLayout()
        hbox.addWidget(self.control)
        hbox.addWidget(self.button)
        self.vbox.addLayout(hbox)
        self.setLayout(self.vbox)

    def widgets(self) -> List[QWidget]:
        """Override to include button."""
        return super().widgets() + [self.button]


class ParamsForm(QWidget):
    """The ParamsForm class is a QWidget that creates controls/inputs for each parameter in the
    provided json file.

  Parameters:
  -----------
    json_file - path of parameters file to be edited.
    load_file - optional path of parameters file to load;
        parameters from this file will be copied over to the json_file.
    width - optional; used to set the width of the form controls.
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
    """
        for key, param in self.params.entries():
            param = Parameter(**param)
            if param.type == 'bool':
                form_input = BoolInput(param)
            elif 'path' in param.type:
                form_input = FileInput(param)
            elif isinstance(param.recommended_values, list):
                form_input = SelectionInput(param)
            else:
                form_input = TextInput(param)
            self.controls[key] = form_input

    def do_layout(self):
        """Layout the form controls."""
        vbox = QVBoxLayout()

        # Add the controls to the grid:
        for param_name, form_input in self.controls.items():
            vbox.addWidget(form_input)

        self.setLayout(vbox)
        self.setFixedWidth(self.width)
        self.show()

    def search(self, text: str):
        """Search for an input. Hides inputs which do not match.
        Searches labels, hints, and values.

        Parameter:
        ----------
          text - text used to search; if empty all inputs are displayed.
         """
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

    def update_parameters(self):
        """Update the parameters from the input values."""

        for param_name, form_input in self.controls.items():
            param = self.params[param_name]
            value = form_input.value()
            if value != param['value']:
                self.params[param_name]['value'] = value


class SearchInput(QWidget):
    """Search input widget. Consists of a text input field and a Clear button.
    Text changes to the input are passed to the on_search action.
    The cancel button clears the input.

    Parameters:
    -----------
        on_search - search function to call. Takes a single str parameter, the
            contents of the text box.
    """

    def __init__(self, on_search, font_size: int = 10):
        super(SearchInput, self).__init__()

        self.on_search = on_search

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(static_text_control(None, label='Search: '))

        self.input = QLineEdit()
        self.input.setStyleSheet(
            f"font-size: {font_size}px; line-height: {font_size}px")
        self.input.textChanged.connect(self._search)
        self.hbox.addWidget(self.input)

        cancel_btn = QPushButton('Clear')
        cancel_btn.setStyleSheet(f"font-size: {font_size}px;")
        cancel_btn.clicked.connect(self._cancel)
        self.hbox.addWidget(cancel_btn)

        self.setLayout(self.hbox)

    def _search(self):
        self.on_search(self.input.text())

    def _cancel(self):
        self.input.clear()
        self.input.repaint()


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

        header_box = QHBoxLayout()
        header_box.addSpacing(5)
        header_box.addWidget(
            static_text_control(self,
                                label=f'Editing: {self.json_file}',
                                size=14,
                                color='dimgray'))
        vbox.addLayout(header_box)

        self.load_msg = static_text_control(self,
                                            label='',
                                            size=12,
                                            color='green')
        vbox.addWidget(self.load_msg)
        self.load_msg.setVisible(False)
        vbox.addWidget(SearchInput(self.search_form))

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

        load_button = QPushButton('Load')
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
            self.load_msg.setText(
                f'Loaded from: {file_name}. Click the Save button to persist these changes.'
            )
            self.load_msg.setVisible(True)
            self.form = ParamsForm(json_file=self.json_file,
                                   load_file=file_name,
                                   width=self.size[0])
            self.form_panel.setWidget(self.form)
            self.repaint()

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
