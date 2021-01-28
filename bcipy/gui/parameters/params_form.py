"""GUI form for editing a Parameters file."""
# pylint: disable=E0611
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout,
                             QPushButton, QScrollArea, QVBoxLayout, QWidget)

from bcipy.gui.gui_main import (
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
from bcipy.helpers.parameters import Parameters


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
        self.help_size = 12
        self.help_color = 'darkgray'

        self.params = Parameters(source=self.load_file, cast_values=False)

        self.controls = {}
        self.create_controls()
        self.do_layout()

    def create_controls(self) -> None:
        """Create controls (inputs, labels, etc) for each item in the
    parameters file.
    """
        for key, param in self.params.entries():
            self.controls[key] = self.parameter_input(param)

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
        form_input = type_inputs.get(param['type'],
                                     SelectionInput if has_options else TextInput)
        return form_input(label=param['readableName'],
                          value=param['value'],
                          help_tip=param['helpTip'],
                          options=param['recommended_values'],
                          help_size=self.help_size,
                          help_color=self.help_color)

    def do_layout(self) -> None:
        """Layout the form controls."""
        vbox = QVBoxLayout()

        # Add the controls to the grid:
        for _param_name, form_input in self.controls.items():
            vbox.addWidget(form_input)

        self.setLayout(vbox)
        self.setFixedWidth(self.width)
        self.show()

    def search(self, text: str) -> None:
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
