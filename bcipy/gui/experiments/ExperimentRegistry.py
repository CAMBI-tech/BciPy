import sys
import subprocess

from bcipy.gui.gui_main import BCIGui, app, AlertMessageType

from bcipy.helpers.load import load_experiments, load_fields
from bcipy.helpers.save import save_experiment_data
from bcipy.helpers.system_utils import DEFAULT_EXPERIMENT_PATH, EXPERIMENT_FILENAME


class ExperimentRegistry(BCIGui):
    """Experiment Registry.

    User interface for creating new experiments for use in BCInterface.py.
    """

    padding = 100
    btn_height = 50
    default_text = '...'
    alert_title = 'Experiment Registry Alert'
    experiment_fields = []

    def __init__(self, *args, **kwargs):
        super(ExperimentRegistry, self).__init__(*args, **kwargs)

        # Structure of an experiment:
        #   { name: { fields : {name: '', required: bool}, summary: '' } }
        self.update_experiment_data()

        # These are set in the build_inputs and represent text inputs from the user
        self.name_input = None
        self.summary_input = None
        self.field_input = None

        self.fields = []
        self.name = None
        self.summary = None

        self.show_gui()
        self.update_field_list()

    def build_text(self) -> None:
        """Build Text.

        Build all static text needed for the UI.
        Positions are relative to the height / width of the UI defined in start_app.
        """
        text_x = 25
        text_y = 70
        font_size = 18
        self.add_static_textbox(
            text='Create BciPy Experiment',
            position=[self.width / 2 - self.padding, 0],
            size=[300, 100],
            background_color='black',
            text_color='white',
            font_size=22
        )
        self.add_static_textbox(
            text='Name',
            position=[text_x, text_y],
            size=[200, 50],
            background_color='black',
            text_color='white',
            font_size=font_size)
        text_y += self.padding
        self.add_static_textbox(
            text='Summary',
            position=[text_x, text_y],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=font_size)
        text_y += self.padding
        self.add_static_textbox(
            text='Fields',
            position=[text_x, text_y],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=font_size)
        text_y += self.padding
        self.add_static_textbox(
            text='Registered fields',
            position=[text_x, text_y],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=font_size)

    def build_inputs(self) -> None:
        """Build Inputs.

        Build all text entry inputs for the UI.
        """
        input_x = 50
        input_y = 120
        input_size = [280, 25]
        self.name_input = self.add_combobox(
            position=[input_x, input_y],
            size=input_size,
            items=[self.default_text],
            editable=True,
            background_color='white',
            text_color='black')

        input_y += self.padding
        self.summary_input = self.add_combobox(
            position=[input_x, input_y],
            size=input_size,
            items=[self.default_text],
            editable=True,
            background_color='white',
            text_color='black')

        input_y += self.padding
        self.field_input = self.add_combobox(
            position=[input_x, input_y],
            size=input_size,
            items=self.fields,
            editable=False,
            background_color='white',
            text_color='black')

    def build_buttons(self):
        """Build Buttons.

        Build all buttons necessary for the UI. Define their action on click using the named argument action.
        """
        btn_create_x = self.width - self.padding
        btn_create_y = self.height - 75
        size = 150
        self.add_button(
            message='Create Experiment', position=[btn_create_x - (size / 2), btn_create_y],
            size=[size, self.btn_height],
            background_color='green',
            action=self.create_experiment,
            text_color='white')

        btn_field_x = (self.width / 2) + 150
        btn_field_y = 310
        # create field
        self.add_button(
            message='+',
            position=[btn_field_x, btn_field_y],
            size=[40, self.btn_height - 10],
            background_color='green',
            action=self.create_field,
            text_color='white'
        )

        # remove field

        # add field
        self.add_button(
            message='Register',
            position=[btn_field_x - 75, btn_field_y],
            size=[60, self.btn_height - 10],
            background_color='grey',
            action=self.add_field,
            text_color='white'
        )

    def create_experiment(self) -> None:
        """Create Experiment.

        After inputing all required fields, verified by check_input, add it to the experiment list and save it.
        """
        if self.check_input():
            self.add_experiment()
            self.save_experiments()
            self.throw_alert_message(
                title=self.alert_title,
                message='Experiment saved successfully! Please exit window or create another experiment!',
                message_type=AlertMessageType.INFO,
                okay_to_exit=True
            )
            self.update_experiment_data()

    def update_experiment_data(self):
        """Update Experiment Data.

        Fetches the experiments and extracts the registered names.
        """
        self.experiments = load_experiments()
        self.experiment_names = self.experiments.keys()

    def add_experiment(self) -> None:
        """Add Experiment:

        Add a new experiment to the dict of experiments. It follows the format:
             { name: { fields : {name: '', required: bool}, summary: '' } }
        """
        self.experiments[self.name] = {
            'fields': self.experiment_fields,
            'summary': self.summary
        }

    def save_experiments(self) -> None:
        """Save Experiment.

        Save the experiments registered to the correct path as pulled from system_utils.
        """
        # add fields to the experiment
        save_experiment_data(self.experiments, DEFAULT_EXPERIMENT_PATH, EXPERIMENT_FILENAME)

    def create_field(self) -> None:
        """Create Field.

        Launch to FieldRegistry to create a new field for experiments.
        """
        subprocess.call(
            f'python bcipy/gui/experiments/FieldRegistry.py',
            shell=True)

        self.update_field_list()

    def add_field(self) -> None:
        """Add Field.

        Functionality to add fields to the newly created experiment. It will ensure no duplicates are addded.
        """
        # get the current field value and compute a list of field names already added
        field = self.field_input.currentText()
        registered_fields = [name for field in self.experiment_fields for name in field.keys()]

        # if the field selected is already registered throw an alert to the user
        if field in registered_fields:
            return self.throw_alert_message(
                title=self.alert_title,
                message='Field already registered in the experiment!',
                message_type=AlertMessageType.INFO,
                okay_to_exit=True
            )

        # else add the field!
        self.experiment_fields.append(
            {
                field:
                    {
                        'required': 'false'
                    }

            }
        )

        self.update_registered_fields()

    def update_registered_fields(self) -> None:
        # TODO
        pass

    def update_field_list(self) -> None:
        """Updates the field_input combo box with a list of fields. """

        self.field_input.clear()
        self.field_input.addItem(ExperimentRegistry.default_text)
        self.fields = [item for item in load_fields()]
        self.field_input.addItems(self.fields)

    def build_assets(self) -> None:
        """Build Assets.

        Define the assets to build in the UI.
        """
        self.build_inputs()
        self.build_text()
        self.build_buttons()

    def check_input(self) -> bool:
        """Check Input.

        Checks to make sure user has input all required fields. Currently, only name and summary are required.
        """
        self.name = self.name_input.currentText()
        self.summary = self.summary_input.currentText()
        try:
            if self.name == ExperimentRegistry.default_text:
                self.throw_alert_message(
                    title=self.alert_title,
                    message='Please add an Experiment Name!',
                    message_type=AlertMessageType.WARN,
                    okay_to_exit=True)
                return False
            if self.name in self.experiment_names:
                self.throw_alert_message(
                    title=self.alert_title,
                    message=(
                        'Experiment name already registered. \n'
                        'Please use a unique Experiment name! \n'
                        f'Registed names: {self.experiment_names}'
                    ),
                    message_type=AlertMessageType.WARN,
                    okay_to_exit=True)
                return False
            if self.summary == ExperimentRegistry.default_text:
                self.throw_alert_message(
                    title=self.alert_title,
                    message='Please add an Experiment Summary!',
                    message_type=AlertMessageType.WARN,
                    okay_to_exit=True)
                return False
        except Exception as e:
            self.throw_alert_message(
                title=self.alert_title,
                message=f'Error, {e}',
                message_type=AlertMessageType.CRIT,
                okay_to_exit=True)
            return False
        return True


def start_app() -> None:
    """Start Experiment Registry."""
    bcipy_gui = app(sys.argv)
    ex = ExperimentRegistry(
        title='Experiment Registry',
        height=600,
        width=550,
        background_color='black')

    sys.exit(bcipy_gui.exec_())


if __name__ == '__main__':
    start_app()
