"""GUI form for collecting experimental field data."""
# pylint: disable=E0611

import sys

from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from bcipy.gui.main import (
    AlertMessageType,
    AlertMessageResponse,
    AlertResponse,
    MessageBox,
    app,
    BoolInput,
    DirectoryInput,
    FileInput,
    FloatInput,
    FormInput,
    IntegerInput,
    TextInput
)
from bcipy.helpers.load import load_experiments, load_fields
from bcipy.helpers.save import save_experiment_field_data


class ExperimentFieldCollection(QWidget):
    """Experiment Field Collection.

    Given an experiment with fields to be collected, this UI can be used to collect data in the correct format
        and require fields which are noted as such in the experiment.
    """
    field_data: List[tuple] = []
    field_inputs: List[FormInput] = []
    type_inputs = {
        'int': IntegerInput,
        'float': FloatInput,
        'bool': BoolInput,
        'filepath': FileInput,
        'directorypath': DirectoryInput
    }
    require_mark = '*'
    alert_timeout = 10
    save_data = {}

    def __init__(self, title: str, width: int, height: int, experiment_name: str, save_path: str, file_name: str):
        super().__init__()

        self.experiment_name = experiment_name
        self.experiment = load_experiments()[experiment_name]
        self.save_path = save_path
        self.file_name = file_name
        self.help_size = 12
        self.help_color = 'darkgray'
        self.width = width
        self.height = height
        self.title = title

        self.fields = load_fields()
        self.build_field_data()
        self.build_assets()
        self.do_layout()

    def do_layout(self) -> None:
        """Layout the form controls."""
        vbox = QVBoxLayout()

        # Add the controls to the grid:
        for form_input in self.field_inputs:
            vbox.addWidget(form_input)

        self.setLayout(vbox)
        self.setFixedWidth(self.width)
        self.show()

    def build_form(self) -> None:
        """Build Form.

        Loop over the field data and create UI field inputs for data collection.
        """
        for field_name, field_type, required, help_text in self.field_data:
            self.field_inputs.append(self.field_input(field_name, field_type, help_text, required))

    def field_input(self, field_name: str, field_type: str, help_tip: str, required: bool) -> FormInput:
        """Field Input.

        Construct a FormInput for the given field based on its python type and other
        attributes.
        """

        form_input = self.type_inputs.get(field_type, TextInput)

        if required:
            field_name += self.require_mark
        return form_input(
            label=field_name,
            value='',
            help_tip=help_tip,
            help_size=self.help_size,
            help_color=self.help_color)

    def build_assets(self) -> None:
        """Build Assets.

        Build any needed assets for the Experiment Field Widget. Currently, only a form is needed.
        """
        self.build_form()

    def check_input(self) -> bool:
        """Check Input.

        Ensure that any fields that require input have data!
        """
        for field in self.field_inputs:
            _input = field.value()
            name = field.label
            if self.require_mark in field.label and not _input:
                self.throw_alert_message(
                    title='BciPy Alert',
                    message=f'Required field {name.strip(self.require_mark)} must be filled out!',
                    message_type=AlertMessageType.CRIT,
                    message_response=AlertMessageResponse.OCE,
                    message_timeout=self.alert_timeout)
                return False
        return True

    def build_field_data(self) -> None:
        """Build Field Data.

        Using the fields defined in the experiment, fetch the other attributes of the field. It will be stored in
            self.field_data as a list of tuples (name, field type, required, help text).
        """
        for field in self.experiment['fields']:
            # the field name and requirement
            for name, required in field.items():
                # help text and type
                field_data = self.fields[name]
                self.field_data.append(
                    (name.title(), field_data['type'], self.map_to_bool(required['required']), field_data['help_text'])
                )

    def map_to_bool(self, string_boolean: str) -> bool:
        """Map To Boolean.

        All data is loaded from json ("true"/"false"). This method will return a python boolean (True/False).
        """
        if string_boolean == 'true':
            return True
        elif string_boolean == 'false':
            return False
        raise Exception(f'Unsupported boolean value {string_boolean}')

    def save(self) -> None:
        if self.check_input():
            self.build_save_data()
            self.write_save_data()

    def build_save_data(self) -> None:
        """Build Save Data."""
        try:
            for field in self.field_inputs:
                _input = field.cast_value()
                name = field.label.strip(self.require_mark)
                self.save_data[name] = _input
        except ValueError as e:
            self.throw_alert_message(
                title='Error',
                message=f'Error saving data. Invalid value provided. \n {e}',
                message_type=AlertMessageType.WARN,
                message_response=AlertMessageResponse.OCE,
                message_timeout=self.alert_timeout
            )

    def write_save_data(self) -> None:
        save_experiment_field_data(self.save_data, self.save_path, self.file_name)
        self.throw_alert_message(
            title='Success',
            message=(
                f'Data successfully written to: \n\n{self.save_path}/{self.file_name} \n\n\n'
                'Please wait or close this window to start the task!'),
            message_type=AlertMessageType.INFO,
            message_response=AlertMessageResponse.OCE,
            message_timeout=self.alert_timeout,
        )

    def throw_alert_message(self,
                            title: str,
                            message: str,
                            message_type: AlertMessageType = AlertMessageType.INFO,
                            message_response: AlertMessageResponse = AlertMessageResponse.OTE,
                            message_timeout: float = 0) -> MessageBox:
        """Throw Alert Message."""

        msg = MessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(message_type.value)
        msg.setTimeout(message_timeout)

        if message_response is AlertMessageResponse.OTE:
            msg.setStandardButtons(AlertResponse.OK.value)
        elif message_response is AlertMessageResponse.OCE:
            msg.setStandardButtons(AlertResponse.OK.value | AlertResponse.CANCEL.value)

        return msg.exec_()


class MainPanel(QWidget):
    """Main GUI window.

    Parameters:
    -----------
      title: window title
      width: window width
      height: window height
      experiment_name: name of the experiment to collect field data for
      save_path: where to save the collected field data
      file_name: name of the file to write with collected field_data
    """

    def __init__(self, title: str, width: int, height: int, experiment_name: str, save_path: str, file_name: str):
        super().__init__()
        self.title = title
        self.width = width
        self.height = height

        self.form = ExperimentFieldCollection(title, width, height, experiment_name, save_path, file_name)
        self.initUI()

    def initUI(self):
        """Initialize the UI"""
        vbox = QVBoxLayout()

        self.form_panel = QScrollArea()
        self.form_panel.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.form_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.form_panel.setWidgetResizable(True)
        self.form_panel.setFixedWidth(self.width)
        self.form_panel.setWidget(self.form)

        vbox.addWidget(self.form_panel)
        vbox.addSpacing(5)

        self.setLayout(vbox)
        self.setFixedHeight(self.height)
        self.setWindowTitle(self.title)

        control_box = QHBoxLayout()
        control_box.addStretch()
        save_button = QPushButton('Save')
        save_button.setFixedWidth(80)
        save_button.clicked.connect(self.save)
        control_box.addWidget(save_button)
        vbox.addLayout(control_box)
        self.show()

    def save(self):
        self.form.save()
        self.close()


def start_app() -> None:
    """Start Experiment Field Collection."""
    import argparse
    from bcipy.config import DEFAULT_EXPERIMENT_ID, EXPERIMENT_DATA_FILENAME
    from bcipy.helpers.validate import validate_experiment, validate_field_data_written

    parser = argparse.ArgumentParser()

    # experiment_name
    parser.add_argument('-p', '--path', default='.',
                        help='Path to save collected field data to in json format')
    parser.add_argument('-e', '--experiment', default=DEFAULT_EXPERIMENT_ID,
                        help='Select a valid experiment to run the task for this user')
    parser.add_argument('-f', '--filename', default=EXPERIMENT_DATA_FILENAME,
                        help='Provide a json filename to write the field data to. Ex, experiment_data.json')
    parser.add_argument('-v', '--validate', default=False,
                        help='Whether or not to validate the experiment before proceeding to data collection.')

    args = parser.parse_args()

    experiment_name = args.experiment
    validate = args.validate

    if validate:
        validate_experiment(experiment_name)
        print('Experiment valid!')

    save_path = args.path
    file_name = args.filename

    bcipy_gui = app(sys.argv)

    ex = MainPanel(
        title='Experiment Field Collection',
        height=250,
        width=600,
        experiment_name=experiment_name,
        save_path=save_path,
        file_name=file_name
    )
    bcipy_gui.exec_()

    if validate:
        if validate_field_data_written(save_path, file_name):
            print('Field data successfully written!')
        else:
            raise Exception(f'Field data not written to {save_path}/{file_name}')

    sys.exit()


if __name__ == '__main__':
    start_app()
