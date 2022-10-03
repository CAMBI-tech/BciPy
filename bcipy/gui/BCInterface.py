import subprocess
import sys

from typing import List

from bcipy.config import BCIPY_ROOT, DEFAULT_PARAMETERS_PATH, STATIC_IMAGES_PATH
from bcipy.gui.main import (
    AlertMessageResponse,
    AlertMessageType,
    AlertResponse,
    app,
    BCIGui,
    contains_special_characters,
    contains_whitespaces,
    invalid_length,
)
from bcipy.helpers.load import load_json_parameters, load_experiments, copy_parameters, load_users
from bcipy.task import TaskType


class BCInterface(BCIGui):
    """BCI Interface.

    Main interface for execution of BciPy experiments and tasks. Additionally, quick access to parameter
        editing and loading, and offline analysis execution.
    """

    tasks = TaskType.list()

    default_text = '...'
    padding = 30
    btn_height = 40
    max_length = 25
    min_length = 1
    timeout = 3

    def __init__(self, *args, **kwargs):
        super(BCInterface, self).__init__(*args, **kwargs)
        self.parameter_location = DEFAULT_PARAMETERS_PATH

        self.parameters = load_json_parameters(self.parameter_location,
                                               value_cast=True)

        # These are set in the build_inputs and represent text inputs from the user
        self.user_input = None
        self.experiment_input = None
        self.task_input = None

        # These represent the current user, experiment, and task selected in the gui
        self.user = None
        self.experiment = None
        self.task = None

        # user names available in the dropdown menu
        self.users = []

        # setup a timer to prevent double clicking in gui
        self.disable = False
        self.timer.timeout.connect(self._disable_action)

        self.task_start_timeout = self.timeout
        self.button_timeout = self.timeout

        self.autoclose = False
        self.alert = True
        self.static_font_size = 24

        self.user_id_validations = [
            (invalid_length(min=self.min_length, max=self.max_length),
             f'User ID must contain between {self.min_length} and {self.max_length} alphanumeric characters.'),
            (contains_whitespaces, 'User ID cannot contain white spaces'),
            (contains_special_characters, 'User ID cannot contain special characters')
        ]

    def build_buttons(self) -> None:
        """Build Buttons.

        Build all buttons necessary for the UI. Define their action on click using the named argument action.
        """
        self.add_button(
            message='Load Parameters', position=[self.padding, 450],
            size=[110, self.btn_height], background_color='white',
            action=self.select_parameters)

        self.add_button(
            message='Edit Parameters', position=[self.padding + 115, 450],
            size=[105, self.btn_height], background_color='white',
            action=self.edit_parameters)

        btn_auc_width = 100
        btn_auc_x = self.padding + 225
        self.add_button(
            message='Calculate AUC', position=(btn_auc_x, 450),
            size=(btn_auc_width, self.btn_height), background_color='white',
            action=self.offline_analysis)

        btn_start_width = 200
        btn_start_x = self.width - (self.padding + btn_start_width)
        self.add_button(
            message='Start Experiment Session', position=[btn_start_x, 450],
            size=[btn_start_width, self.btn_height],
            background_color='green',
            action=self.start_experiment,
            text_color='white')

        self.add_button(
            message='+',
            position=[self.width - self.padding - 200, 260],
            size=[40, self.btn_height - 10],
            background_color='green',
            action=self.create_experiment,
            text_color='white'
        )

    def create_experiment(self) -> None:
        """Create Experiment.

        Launch the experiment registry which will be used to add new experiments for selection in the GUI.
        """
        if not self.action_disabled():
            subprocess.call(
                f'python {BCIPY_ROOT}/gui/experiments/ExperimentRegistry.py',
                shell=True)

            self.update_experiment_list()

    def update_user_list(self, refresh=True) -> None:
        """Updates the user_input combo box with a list of user ids based on the
        data directory configured in the current parameters."""
        # if refresh is True, then we need to clear the list and add the default text
        if refresh:
            self.user_input.clear()
            self.user_input.addItem(BCInterface.default_text)

        # load the users from the data directory and check if they have already been added to the dropdown
        users = load_users(self.parameters['data_save_loc'])
        for user in users:
            if user not in self.users:
                self.user_input.addItem(user)
                self.users.append(user)

    def update_experiment_list(self) -> None:
        """Updates the experiment_input combo box with a list of experiments based on the
        data directory configured in the current parameters."""

        self.experiment_input.clear()
        self.experiment_input.addItem(BCInterface.default_text)
        self.experiment_input.addItems(self.load_experiments())

    def update_task_list(self) -> None:
        """Updates the task_input combo box with a list of tasks ids based on what
        is available in the Task Registry"""

        self.task_input.clear()
        self.task_input.addItem(BCInterface.default_text)
        self.task_input.addItems(self.tasks)

    def build_inputs(self) -> None:
        """Build Inputs.

        Build all inputs needed for BCInterface.
        """
        input_x = 170
        input_y = 160
        self.user_input = self.add_combobox(
            position=[input_x, input_y],
            size=[280, 25],
            items=[],
            editable=True,
            background_color='white',
            text_color='black')

        self.update_user_list()

        input_y += 100
        self.experiment_input = self.add_combobox(
            position=[input_x, input_y],
            size=[280, 25],
            items=[],
            editable=False,
            background_color='white',
            text_color='black')

        self.update_experiment_list()

        input_y += 100
        self.task_input = self.add_combobox(
            position=[input_x, input_y],
            size=[280, 25],
            items=[],
            editable=False,
            background_color='white',
            text_color='black')

        self.update_task_list()

    def build_text(self) -> None:
        """Build Text.

        Build all static text needed for the UI.
        Positions are relative to the height / width of the UI defined in start_app.
        """
        self.add_static_textbox(
            text='BCInterface',
            position=[275, 0],
            size=[200, 50],
            background_color='black',
            text_color='white',
            font_size=30)

        text_x = 145
        self.add_static_textbox(
            text='User',
            position=[text_x, 105],
            size=[200, 50],
            background_color='black',
            text_color='white',
            font_size=self.static_font_size)
        self.add_static_textbox(
            text='Experiment',
            position=[text_x, 205],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=self.static_font_size)
        self.add_static_textbox(
            text='Task',
            position=[text_x, 305],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=self.static_font_size)

    def build_images(self) -> None:
        """Build Images.

        Build add images needed for the UI. In this case, the OHSU and NEU logos.
        """
        self.add_image(
            path=f'{STATIC_IMAGES_PATH}/gui/ohsu.png', position=[self.padding, 0], size=100)
        self.add_image(
            path=f'{STATIC_IMAGES_PATH}/gui/neu.png', position=[self.width - self.padding - 110, 0], size=100)

    def build_assets(self) -> None:
        """Build Assets.

        Define the assets to build in the UI.
        """
        self.build_buttons()
        self.build_inputs()
        self.build_text()
        self.build_images()

    def set_parameter_location(self, path: str) -> None:
        """Sets the parameter_location to the given path. Reloads the parameters
        and updates any GUI widgets that are populated based on these params."""
        self.parameter_location = path
        self.parameters = load_json_parameters(self.parameter_location,
                                               value_cast=True)
        self.update_user_list(refresh=False)

    def select_parameters(self) -> None:
        """Select Parameters.

        Opens a dialog to select the parameters.json configuration to use.
        """

        response = self.get_filename_dialog(message='Select parameters file',
                                            file_type='JSON files (*.json)')
        if response:
            self.set_parameter_location(response)
            # If outdated, prompt to merge with the current defaults
            default_parameters = load_json_parameters(DEFAULT_PARAMETERS_PATH,
                                                      value_cast=True)
            if self.parameters.add_missing_items(default_parameters):
                save_response = self.throw_alert_message(
                    title='BciPy Alert',
                    message='The selected parameters file is out of date.'
                            'Would you like to update it with the latest options?',
                    message_type=AlertMessageType.INFO,
                    message_response=AlertMessageResponse.OCE)

                if save_response == AlertResponse.OK.value:
                    self.parameters.save()

    def edit_parameters(self) -> None:
        """Edit Parameters.

        Prompts for a parameters.json file to use. If the default parameters are selected, a copy is used.
        Note that any edits to the parameters file will not be applied to this GUI until the parameters
        are reloaded.
        """
        if not self.action_disabled():
            if self.parameter_location == DEFAULT_PARAMETERS_PATH:
                # Don't allow the user to overwrite the defaults
                response = self.throw_alert_message(
                    title='BciPy Alert',
                    message='The default parameters.json cannot be overridden. A copy will be used.',
                    message_type=AlertMessageType.INFO,
                    message_response=AlertMessageResponse.OCE)

                if response == AlertResponse.OK.value:
                    self.parameter_location = copy_parameters()
                else:
                    return None

            subprocess.call(
                f'python {BCIPY_ROOT}/gui/parameters/params_form.py -p {self.parameter_location}',
                shell=True)

    def check_input(self) -> bool:
        """Check Input.

        Checks to make sure user has input all required fields.
        """

        # Update based on current inputs
        self.user = self.user_input.currentText()
        self.experiment = self.experiment_input.currentText()
        self.task = self.task_input.currentText()

        # Check the set values are different than defaults
        try:
            if not self.check_user_id():
                return False
            if self.experiment == BCInterface.default_text:
                self.throw_alert_message(
                    title='BciPy Alert',
                    message='Please select or create an Experiment',
                    message_type=AlertMessageType.INFO,
                    message_response=AlertMessageResponse.OTE)
                return False
            if self.task == BCInterface.default_text:
                self.throw_alert_message(
                    title='BciPy Alert',
                    message='Please select a Task',
                    message_type=AlertMessageType.INFO,
                    message_response=AlertMessageResponse.OTE)
                return False
        except Exception as e:
            self.throw_alert_message(
                title='BciPy Alert',
                message=f'Error, {e}',
                message_type=AlertMessageType.CRIT,
                message_response=AlertMessageResponse.OTE)
            return False
        return True

    def check_user_id(self) -> bool:
        """Check User ID

        User ID must meet the following requirements:

        1. Maximum length of self.max_length alphanumeric characters
        2. Minimum length of at least self.min_length alphanumeric character
        3. No special characters
        4. No spaces
        """
        # Check the user id set is different than the default text
        if self.user == BCInterface.default_text:
            self.throw_alert_message(
                title='BciPy Alert',
                message='Please input a User ID',
                message_type=AlertMessageType.INFO,
                message_response=AlertMessageResponse.OTE)
            return False
        # Loop over defined user validations and check for error conditions
        for validator in self.user_id_validations:
            (invalid, error_message) = validator
            if invalid(self.user):
                self.throw_alert_message(
                    title='BciPy Alert',
                    message=error_message,
                    message_type=AlertMessageType.INFO,
                    message_response=AlertMessageResponse.OTE
                )
                return False
        return True

    def load_experiments(self) -> List[str]:
        """Load experiments

        Loads experiments registered in the DEFAULT_EXPERIMENT_PATH.
        """
        return load_experiments().keys()

    def start_experiment(self) -> None:
        """Start Experiment Session.

        Using the inputs gathers, check for validity using the check_input method, then launch the experiment using a
            command to bcipy main and subprocess.
        """
        if self.check_input() and not self.action_disabled():
            self.throw_alert_message(
                title='BciPy Alert',
                message='Task Starting ...',
                message_type=AlertMessageType.INFO,
                message_response=AlertMessageResponse.OTE,
                message_timeout=self.task_start_timeout)
            cmd = (
                f'bcipy -e "{self.experiment}" '
                f'-u "{self.user}" -t "{self.task}" -p "{self.parameter_location}"'
            )
            if self.alert:
                cmd += ' -a'
            subprocess.Popen(cmd, shell=True)

            if self.autoclose:
                self.close()

    def offline_analysis(self) -> None:
        """Offline Analysis.

        Run offline analysis as a script in a new process.
        """
        if not self.action_disabled():
            cmd = f'python {BCIPY_ROOT}/signal/model/offline_analysis.py --alert --p "{self.parameter_location}"'
            subprocess.Popen(cmd, shell=True)

    def action_disabled(self) -> bool:
        """Action Disabled.

        Method to check whether another action can take place. If not disabled, it will allow the action and
        start a timer that will disable actions until self.timeout (seconds) has occured.

        Note: the timer is registed with the private method self._disable_action, which when self.timeout has
        been reached, resets self.disable and corresponding timeouts.
        """
        if self.disable:
            return True
        else:
            self.disable = True
            # set the update time to every 1000ms
            self.timer.start(1000)
            return False

    def _disable_action(self) -> bool:
        """Disable Action.

        A private method to register with a BCIGui.timer after setting self.button_timeout.
        """
        if self.button_timeout > 0:
            self.disable = True
            self.button_timeout -= 1
            return self.disable

        self.timer.stop()
        self.disable = False
        self.button_timeout = self.timeout
        return self.disable


def start_app() -> None:
    """Start BCIGui."""
    bcipy_gui = app(sys.argv)
    ex = BCInterface(
        title='Brain Computer Interface',
        height=550,
        width=750,
        background_color='black')

    ex.show_gui()

    sys.exit(bcipy_gui.exec_())


if __name__ == '__main__':
    start_app()
