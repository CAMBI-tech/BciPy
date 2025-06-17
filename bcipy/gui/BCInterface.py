"""BCInterface module.

This module provides the main graphical user interface for BciPy experiments,
including task execution, parameter management, and offline analysis capabilities.
"""

import logging
import subprocess
import sys
from typing import List, Optional, Dict, Any, Tuple, Callable

from bcipy.config import (BCIPY_ROOT, DEFAULT_PARAMETERS_PATH,
                          PROTOCOL_LOG_FILENAME, STATIC_IMAGES_PATH)
from bcipy.gui.main import (AlertMessageResponse, AlertMessageType,
                            AlertResponse, BCIGui, app,
                            contains_special_characters, contains_whitespaces,
                            invalid_length)
from bcipy.io.load import (copy_parameters, load_experiments,
                           load_json_parameters, load_users)
from bcipy.task import TaskRegistry

logger = logging.getLogger(PROTOCOL_LOG_FILENAME)


class BCInterface(BCIGui):
    """BCI Interface.

    Main interface for execution of BciPy experiments and tasks. Provides quick access to parameter
    editing and loading, and offline analysis execution.

    Attributes:
        tasks (List[str]): List of available tasks from TaskRegistry.
        default_text (str): Default text for input fields.
        padding (int): Padding value for UI elements.
        btn_height (int): Height of buttons.
        btn_width (int): Width of buttons.
        max_length (int): Maximum length for user IDs.
        min_length (int): Minimum length for user IDs.
        timeout (int): Timeout value for actions.
        font (str): Font family for UI elements.
        parameter_location (str): Path to parameters file.
        parameters (Dict[str, Any]): Loaded parameters.
        user_input (Optional[Any]): User input field.
        experiment_input (Optional[Any]): Experiment input field.
        task_input (Optional[Any]): Task input field.
        user (Optional[str]): Selected user.
        experiment (Optional[str]): Selected experiment.
        task (Optional[str]): Selected task.
        users (List[str]): List of available users.
        disable (bool): Flag to prevent double-clicking.
        task_start_timeout (int): Timeout for task start.
        button_timeout (int): Timeout for button actions.
        autoclose (bool): Whether to auto-close after task completion.
        alert (bool): Whether to show alerts.
        static_font_size (int): Font size for static text.
        user_id_validations (List[Tuple[Callable[[str], bool], str]]): User ID validation rules.
    """

    tasks = TaskRegistry().list()

    default_text = '...'
    padding = 20
    btn_height = 40
    btn_width = 100
    max_length = 25
    min_length = 1
    timeout = 3
    font = 'Courier New'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the BCInterface.

        Args:
            *args: Positional arguments passed to BCIGui.
            **kwargs: Keyword arguments passed to BCIGui.
        """
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
        self.users: List[str] = []

        # setup a timer to prevent double clicking in gui
        self.disable = False
        self.timer.timeout.connect(self._disable_action)

        self.task_start_timeout = self.timeout
        self.button_timeout = self.timeout

        self.autoclose = False
        self.alert = True
        self.static_font_size = 16

        self.user_id_validations = [
            (invalid_length(min=self.min_length, max=self.max_length),
             f'User ID must contain between {self.min_length} and {self.max_length} alphanumeric characters.'),
            (contains_whitespaces, 'User ID cannot contain white spaces'),
            (contains_special_characters, 'User ID cannot contain special characters')
        ]

    def build_buttons(self) -> None:
        """Build all buttons for the UI.

        Creates and configures buttons for loading parameters, editing parameters,
        running offline analysis, starting sessions, and creating experiments.
        """
        self.add_button(
            message='Load', position=[self.padding, 450],
            size=[self.btn_width, self.btn_height], background_color='Plum',
            text_color='black',
            font_family=self.font,
            action=self.select_parameters)

        self.add_button(
            message='Edit', position=[self.padding + self.btn_width + 10, 450],
            size=[self.btn_width, self.btn_height], background_color='LightCoral',
            text_color='black',
            font_family=self.font,
            action=self.edit_parameters)

        btn_auc_x = self.padding + (self.btn_width * 2) + 20
        self.add_button(
            message='Train', position=(btn_auc_x, 450),
            size=(self.btn_width, self.btn_height), background_color='LightSeaGreen',
            text_color='black',
            font_family=self.font,
            action=self.offline_analysis)

        btn_start_width = self.btn_width * 2 + 10
        btn_start_x = self.width - (self.padding + btn_start_width)
        self.add_button(
            message='Start Session', position=[btn_start_x, 440],
            size=[btn_start_width, self.btn_height + 10],
            background_color='green',
            action=self.start_experiment,
            text_color='white',
            font_family=self.font)

        self.add_button(
            message='+',
            position=[self.width - self.padding - 200, 260],
            size=[35, self.btn_height - 10],
            background_color='green',
            action=self.create_experiment,
            text_color='white'
        )

    def create_experiment(self) -> None:
        """Launch the experiment registry.

        Opens the experiment registry interface for adding new experiments
        to the GUI selection.
        """
        if not self.action_disabled():
            subprocess.call(
                f'python {BCIPY_ROOT}/gui/experiments/ExperimentRegistry.py',
                shell=True)

            self.update_experiment_list()

    def update_user_list(self, refresh: bool = True) -> None:
        """Update the user input combo box with available users.

        Args:
            refresh (bool): Whether to clear the list before updating.
        """
        if refresh:
            if self.user_input:
                self.user_input.clear()
                self.user_input.addItem(BCInterface.default_text)

        users = load_users(self.parameters['data_save_loc'])
        for user in users:
            if user not in self.users:
                if self.user_input:
                    self.user_input.addItem(user)
                self.users.append(user)

    def update_experiment_list(self) -> None:
        """Update the experiment input combo box with available experiments."""
        if self.experiment_input:
            self.experiment_input.clear()
            self.experiment_input.addItem(BCInterface.default_text)
            self.experiment_input.addItems(self.load_experiments())

    def update_task_list(self) -> None:
        """Update the task input combo box with available tasks."""
        if self.task_input:
            self.task_input.clear()
            self.task_input.addItem(BCInterface.default_text)
            self.task_input.addItems(self.tasks)

    def build_inputs(self) -> None:
        """Build all input fields for the interface.

        Creates and configures combo boxes for user, experiment, and task selection.
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
        """Build all static text elements for the interface.

        Creates and configures text labels for the interface title and input fields.
        """
        self.add_static_textbox(
            text='BCInterface',
            position=[210, 0],
            size=[250, 50],
            background_color='black',
            text_color='white',
            font_family=self.font,
            font_size=30)

        text_x = 145
        self.add_static_textbox(
            text='User',
            position=[text_x, 105],
            size=[200, 50],
            background_color='black',
            text_color='white',
            font_family=self.font,
            font_size=self.static_font_size)
        self.add_static_textbox(
            text='Experiment',
            position=[text_x, 205],
            size=[300, 50],
            background_color='black',
            font_family=self.font,
            text_color='white',
            font_size=self.static_font_size)
        self.add_static_textbox(
            text='Task',
            position=[text_x, 305],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_family=self.font,
            font_size=self.static_font_size)

    def build_images(self) -> None:
        """Build all image elements for the interface.

        Adds institutional logos to the interface.
        """
        self.add_image(
            path=f'{STATIC_IMAGES_PATH}/gui/ohsu.png', position=[self.padding, 0], size=100)
        self.add_image(
            path=f'{STATIC_IMAGES_PATH}/gui/neu.png', position=[self.width - self.padding - 110, 0], size=100)

    def build_assets(self) -> None:
        """Build all UI assets.

        Calls methods to build buttons, inputs, text, and images.
        """
        self.build_buttons()
        self.build_inputs()
        self.build_text()
        self.build_images()

    def set_parameter_location(self, path: str) -> None:
        """Set the parameter file location and update the interface.

        Args:
            path (str): Path to the parameters file.
        """
        self.parameter_location = path
        self.parameters = load_json_parameters(self.parameter_location,
                                               value_cast=True)
        self.update_user_list(refresh=False)

    def select_parameters(self) -> None:
        """Open a dialog to select the parameters configuration file.

        Prompts the user to select a parameters.json file and handles
        updating outdated parameter files.
        """
        response = self.get_filename_dialog(message='Select parameters file',
                                            file_type='JSON files (*.json)')
        if response:
            self.set_parameter_location(response)
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
        """Open the parameter editor.

        Prompts for a parameters.json file to edit. If the default parameters
        are selected, a copy is used instead.
        """
        if not self.action_disabled():
            if self.parameter_location == DEFAULT_PARAMETERS_PATH:
                response = self.throw_alert_message(
                    title='BciPy Alert',
                    message='The default parameters.json cannot be overridden. A copy will be used.',
                    message_type=AlertMessageType.INFO,
                    message_response=AlertMessageResponse.OCE)

                if response == AlertResponse.OK.value:
                    self.parameter_location = copy_parameters()
                else:
                    return None

            output = subprocess.check_output(
                f'bcipy-params -p "{self.parameter_location}"',
                shell=True)
            if output:
                self.parameter_location = output.decode().strip()

    def check_input(self) -> bool:
        """Check if all required input fields are valid.

        Returns:
            bool: True if all inputs are valid, False otherwise.
        """
        if self.user_input:
            self.user = self.user_input.currentText()
        if self.experiment_input:
            self.experiment = self.experiment_input.currentText()
        if self.task_input:
            self.task = self.task_input.currentText()

        try:
            if not self.check_user_id():
                return False

            if self.experiment == BCInterface.default_text and self.task == BCInterface.default_text:
                self.throw_alert_message(
                    title='BciPy Alert',
                    message='Please select an Experiment or Task for execution',
                    message_type=AlertMessageType.INFO,
                    message_response=AlertMessageResponse.OTE)
                return False
            if self.experiment != BCInterface.default_text and self.task != BCInterface.default_text:
                self.throw_alert_message(
                    title='BciPy Alert',
                    message='Please select only an Experiment or Task',
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
        """Validate the user ID against defined requirements.

        Requirements:
            1. Maximum length of self.max_length alphanumeric characters
            2. Minimum length of at least self.min_length alphanumeric character
            3. No special characters
            4. No spaces

        Returns:
            bool: True if user ID is valid, False otherwise.
        """
        if self.user == BCInterface.default_text:
            self.throw_alert_message(
                title='BciPy Alert',
                message='Please input a User ID',
                message_type=AlertMessageType.INFO,
                message_response=AlertMessageResponse.OTE)
            return False
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
        """Load available experiments from the default experiment path.

        Returns:
            List[str]: List of experiment names.
        """
        return load_experiments().keys()

    def start_experiment(self) -> None:
        """Start an experiment session.

        Validates inputs and launches the experiment using subprocess.
        """
        if self.check_input() and not self.action_disabled():
            self.throw_alert_message(
                title='BciPy Alert',
                message='Task Starting ...',
                message_type=AlertMessageType.INFO,
                message_response=AlertMessageResponse.OTE,
                message_timeout=self.task_start_timeout)
            if self.task != BCInterface.default_text:
                cmd = (
                    f'bcipy '
                    f'-u "{self.user}" -t "{self.task}" -p "{self.parameter_location}"'
                )
            else:
                cmd = (
                    f'bcipy '
                    f'-u "{self.user}" -e "{self.experiment}" -p "{self.parameter_location}"'
                )
            if self.alert:
                cmd += ' -a'
            output = subprocess.run(cmd, shell=True)
            if output.returncode != 0:
                self.throw_alert_message(
                    title='BciPy Alert',
                    message=f'Error: {output.stderr.decode()}',
                    message_type=AlertMessageType.CRIT,
                    message_response=AlertMessageResponse.OTE)

            if self.autoclose:
                self.close()

    def offline_analysis(self) -> None:
        """Run offline analysis in a new process."""
        if not self.action_disabled():
            cmd = f'bcipy-train --alert --p "{self.parameter_location}" -v -s'
            subprocess.Popen(cmd, shell=True)

    def action_disabled(self) -> bool:
        """Check if actions are currently disabled.

        Returns:
            bool: True if actions are disabled, False otherwise.
        """
        if self.disable:
            return True
        else:
            self.disable = True
            self.timer.start(500)
            return False

    def _disable_action(self) -> bool:
        """Handle action disabling timer.

        Returns:
            bool: Current disabled state.
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
    """Start the BCI interface application."""
    bcipy_gui = app(sys.argv)
    ex = BCInterface(
        title='Brain Computer Interface',
        height=550,
        width=700,
        background_color='black')

    ex.show_gui()

    sys.exit(bcipy_gui.exec())


if __name__ == '__main__':
    start_app()
