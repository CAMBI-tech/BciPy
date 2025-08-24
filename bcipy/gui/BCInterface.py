"""BCInterface module.

This module provides the main graphical user interface for BciPy experiments,
including task execution, parameter management, and offline analysis capabilities.
"""

import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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


@dataclass
class UIConfig:
    """Configuration for UI elements.

    Attributes:
        padding (int): Padding value for UI elements.
        btn_height (int): Height of buttons.
        btn_width (int): Width of buttons.
        font (str): Font family for UI elements.
        static_font_size (int): Font size for static text.
    """
    padding: int = 20
    btn_height: int = 40
    btn_width: int = 100
    font: str = 'Courier New'
    static_font_size: int = 16


@dataclass
class UserConfig:
    """Configuration for user-related settings.

    Attributes:
        max_length (int): Maximum length for user IDs.
        min_length (int): Minimum length for user IDs.
        default_text (str): Default text for input fields.
    """
    max_length: int = 25
    min_length: int = 1
    default_text: str = '...'


class BCInterface(BCIGui):
    """BCI Interface.

    Main interface for execution of BciPy experiments and tasks. Provides quick access to parameter
    editing and loading, and offline analysis execution.

    Attributes:
        tasks (List[str]): List of available tasks from TaskRegistry.
        ui_config (UIConfig): UI configuration settings.
        user_config (UserConfig): User-related configuration settings.
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
        user_id_validations (List[Tuple[Callable[[str], bool], str]]): User ID validation rules.
    """

    tasks = TaskRegistry().list()
    ui_config = UIConfig()
    user_config = UserConfig()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the BCInterface.

        Args:
            *args: Positional arguments passed to BCIGui.
            **kwargs: Keyword arguments passed to BCIGui.
        """
        super(BCInterface, self).__init__(*args, **kwargs)
        self.parameter_location = DEFAULT_PARAMETERS_PATH
        self.parameters = load_json_parameters(
            self.parameter_location, value_cast=True)

        # Input fields
        self.user_input = None
        self.experiment_input = None
        self.task_input = None

        # Selected values
        self.user = None
        self.experiment = None
        self.task = None
        self.users: List[str] = []

        # UI state
        self.disable = False
        self.timer.timeout.connect(self._disable_action)
        self.task_start_timeout = 3
        self.button_timeout = 3
        self.autoclose = False
        self.alert = True

        # Initialize user ID validations
        self.user_id_validations = self._init_user_validations()

        # Load tasks
        self.tasks = TaskRegistry().list()
        if not self.tasks:
            logger.warning("No tasks found in TaskRegistry")

    def _init_user_validations(self) -> List[Tuple[Callable[[str], bool], str]]:
        """Initialize user ID validation rules.

        Returns:
            List[Tuple[Callable[[str], bool], str]]: List of validation rules and error messages.
        """
        return [
            (invalid_length(min=self.user_config.min_length, max=self.user_config.max_length),
             f'User ID must contain between {self.user_config.min_length} and {self.user_config.max_length} alphanumeric characters.'),
            (contains_whitespaces, 'User ID cannot contain white spaces'),
            (contains_special_characters, 'User ID cannot contain special characters')
        ]

    def build_buttons(self) -> None:
        """Build all buttons for the UI."""
        self._build_action_buttons()
        self._build_start_button()
        self._build_create_experiment_button()

    def _build_action_buttons(self) -> None:
        """Build the action buttons (Load, Edit, Train)."""
        # Load button
        self.add_button(
            message='Load',
            position=[self.ui_config.padding, 450],
            size=[self.ui_config.btn_width, self.ui_config.btn_height],
            background_color='Plum',
            text_color='black',
            font_family=self.ui_config.font,
            action=self.select_parameters)

        # Edit button
        self.add_button(
            message='Edit',
            position=[self.ui_config.padding +
                      self.ui_config.btn_width + 10, 450],
            size=[self.ui_config.btn_width, self.ui_config.btn_height],
            background_color='LightCoral',
            text_color='black',
            font_family=self.ui_config.font,
            action=self.edit_parameters)

        # Train button
        btn_auc_x = self.ui_config.padding + \
            (self.ui_config.btn_width * 2) + 20
        self.add_button(
            message='Train',
            position=(btn_auc_x, 450),
            size=(self.ui_config.btn_width, self.ui_config.btn_height),
            background_color='LightSeaGreen',
            text_color='black',
            font_family=self.ui_config.font,
            action=self.offline_analysis)

    def _build_start_button(self) -> None:
        """Build the Start Session button."""
        btn_start_width = self.ui_config.btn_width * 2 + 10
        btn_start_x = self.width - (self.ui_config.padding + btn_start_width)
        self.add_button(
            message='Start Session',
            position=[btn_start_x, 440],
            size=[btn_start_width, self.ui_config.btn_height + 10],
            background_color='green',
            action=self.start_experiment,
            text_color='white',
            font_family=self.ui_config.font)

    def _build_create_experiment_button(self) -> None:
        """Build the Create Experiment button."""
        self.add_button(
            message='+',
            position=[self.width - self.ui_config.padding - 200, 260],
            size=[35, self.ui_config.btn_height - 10],
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

    def build_inputs(self) -> None:
        """Build all input fields for the interface."""
        input_x = 170
        input_y = 160

        # User input
        self.user_input = self._build_combobox(
            position=[input_x, input_y],
            editable=True)
        self.update_user_list()

        # Experiment input
        input_y += 100
        self.experiment_input = self._build_combobox(
            position=[input_x, input_y],
            editable=False)
        self.update_experiment_list()

        # Task input
        input_y += 100
        self.task_input = self._build_combobox(
            position=[input_x, input_y],
            editable=False)
        self.update_task_list()

    def _build_combobox(self, position: List[int], editable: bool) -> Any:
        """Build a combo box with standard styling.

        Args:
            position (List[int]): Position coordinates [x, y].
            editable (bool): Whether the combo box is editable.

        Returns:
            Any: The created combo box.
        """
        combo = self.add_combobox(
            position=position,
            size=[280, 25],
            items=[self.user_config.default_text],
            editable=editable,
            background_color='white',
            text_color='black')
        return combo

    def build_text(self) -> None:
        """Build all static text elements for the interface."""
        # Title
        self.add_static_textbox(
            text='BCInterface',
            position=[210, 0],
            size=[250, 50],
            background_color='black',
            text_color='white',
            font_family=self.ui_config.font,
            font_size=30)

        # Labels
        text_x = 145
        labels = [
            ('User', 105),
            ('Experiment', 205),
            ('Task', 305)
        ]

        for text, y_pos in labels:
            self.add_static_textbox(
                text=text,
                position=[text_x, y_pos],
                size=[300, 50],
                background_color='black',
                text_color='white',
                font_family=self.ui_config.font,
                font_size=self.ui_config.static_font_size)

    def build_images(self) -> None:
        """Build all image elements for the interface."""
        # OHSU logo
        self.add_image(
            path=f'{STATIC_IMAGES_PATH}/gui/ohsu.png',
            position=[self.ui_config.padding, 0],
            size=100)

        # NEU logo
        self.add_image(
            path=f'{STATIC_IMAGES_PATH}/gui/neu.png',
            position=[self.width - self.ui_config.padding - 110, 0],
            size=100)

    def build_assets(self) -> None:
        """Build all UI assets."""
        self.build_buttons()
        self.build_inputs()
        self.build_text()
        self.build_images()

    def update_user_list(self, refresh: bool = True) -> None:
        """Update the user input combo box with available users.

        Args:
            refresh (bool): Whether to clear the list before updating.
        """
        if refresh and self.user_input:
            self.user_input.clear()
            self.user_input.addItem(self.user_config.default_text)

        users = load_users(self.parameters['data_save_loc'])
        for user in users:
            if user not in self.users and self.user_input:
                self.user_input.addItem(user)
                self.users.append(user)

    def update_experiment_list(self) -> None:
        """Update the experiment input combo box with available experiments."""
        if self.experiment_input:
            self.experiment_input.clear()
            self.experiment_input.addItem(self.user_config.default_text)
            experiments = self.load_experiments()
            if experiments:
                self.experiment_input.addItems(experiments)

    def update_task_list(self) -> None:
        """Update the task input combo box with available tasks."""
        if self.task_input:
            self.task_input.clear()
            self.task_input.addItem(self.user_config.default_text)
            if self.tasks:
                self.task_input.addItems(self.tasks)

    def set_parameter_location(self, path: str) -> None:
        """Set the parameter file location and update the interface.

        Args:
            path (str): Path to the parameters file.
        """
        self.parameter_location = path
        self.parameters = load_json_parameters(
            self.parameter_location, value_cast=True)
        self.update_user_list(refresh=False)

    def select_parameters(self) -> None:
        """Open a dialog to select the parameters configuration file."""
        response = self.get_filename_dialog(
            message='Select parameters file',
            file_type='JSON files (*.json)')

        if not response:
            return

        self.set_parameter_location(response)
        self._handle_outdated_parameters()

    def _handle_outdated_parameters(self) -> None:
        """Handle outdated parameter files by prompting for updates."""
        default_parameters = load_json_parameters(
            DEFAULT_PARAMETERS_PATH, value_cast=True)
        if not self.parameters.add_missing_items(default_parameters):
            return

        save_response = self.throw_alert_message(
            title='BciPy Alert',
            message='The selected parameters file is out of date. '
                    'Would you like to update it with the latest options?',
            message_type=AlertMessageType.INFO,
            message_response=AlertMessageResponse.OCE)

        if save_response == AlertResponse.OK.value:
            self.parameters.save()

    def edit_parameters(self) -> None:
        """Open the parameter editor."""
        if self.action_disabled():
            return

        if self.parameter_location == DEFAULT_PARAMETERS_PATH:
            if not self._handle_default_parameters():
                return

        self._launch_parameter_editor()

    def _handle_default_parameters(self) -> bool:
        """Handle editing of default parameters.

        Returns:
            bool: True if should continue with editing, False otherwise.
        """
        response = self.throw_alert_message(
            title='BciPy Alert',
            message='The default parameters.json cannot be overridden. A copy will be used.',
            message_type=AlertMessageType.INFO,
            message_response=AlertMessageResponse.OCE)

        if response == AlertResponse.OK.value:
            self.parameter_location = copy_parameters()
            return True
        return False

    def _launch_parameter_editor(self) -> None:
        """Launch the parameter editor process."""
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
        self._update_current_selections()

        try:
            if not self.check_user_id():
                return False

            if not self._validate_experiment_task_selection():
                return False

        except Exception as e:
            self._show_error_alert(str(e))
            return False

        return True

    def _update_current_selections(self) -> None:
        """Update current selections from input fields."""
        if self.user_input:
            self.user = self.user_input.currentText()
        if self.experiment_input:
            self.experiment = self.experiment_input.currentText()
        if self.task_input:
            self.task = self.task_input.currentText()

    def _validate_experiment_task_selection(self) -> bool:
        """Validate experiment and task selections.

        Returns:
            bool: True if selections are valid, False otherwise.
        """
        if self.experiment == self.user_config.default_text and self.task == self.user_config.default_text:
            self.throw_alert_message(
                title='BciPy Alert',
                message='Please select an Experiment or Task for execution',
                message_type=AlertMessageType.INFO,
                message_response=AlertMessageResponse.OTE)
            return False

        if self.experiment != self.user_config.default_text and self.task != self.user_config.default_text:
            self.throw_alert_message(
                title='BciPy Alert',
                message='Please select only an Experiment or Task',
                message_type=AlertMessageType.INFO,
                message_response=AlertMessageResponse.OTE)
            return False

        return True

    def _show_error_alert(self, error_message: str) -> None:
        """Show an error alert message.

        Args:
            error_message (str): The error message to display.
        """
        self.throw_alert_message(
            title='BciPy Alert',
            message=f'Error, {error_message}',
            message_type=AlertMessageType.CRIT,
            message_response=AlertMessageResponse.OTE)

    def check_user_id(self) -> bool:
        """Validate the user ID against defined requirements.

        Returns:
            bool: True if user ID is valid, False otherwise.
        """
        if not self.user or self.user == self.user_config.default_text:
            self.throw_alert_message(
                title='BciPy Alert',
                message='Please input a User ID',
                message_type=AlertMessageType.INFO,
                message_response=AlertMessageResponse.OTE)
            return False

        for validator, error_message in self.user_id_validations:
            if validator(str(self.user)):  # Ensure user is treated as string
                self.throw_alert_message(
                    title='BciPy Alert',
                    message=error_message,
                    message_type=AlertMessageType.INFO,
                    message_response=AlertMessageResponse.OTE)
                return False

        return True

    def load_experiments(self) -> List[str]:
        """Load available experiments from the default experiment path.

        Returns:
            List[str]: List of experiment names.
        """
        return load_experiments().keys()

    def start_experiment(self) -> None:
        """Start an experiment session."""
        if not (self.check_input() and not self.action_disabled()):
            return

        self._show_starting_alert()
        cmd = self._build_experiment_command()
        self._execute_experiment(cmd)

    def _show_starting_alert(self) -> None:
        """Show the task starting alert."""
        self.throw_alert_message(
            title='BciPy Alert',
            message='Task Starting ...',
            message_type=AlertMessageType.INFO,
            message_response=AlertMessageResponse.OTE,
            message_timeout=self.task_start_timeout)

    def _build_experiment_command(self) -> str:
        """Build the experiment command.

        Returns:
            str: The command to execute.
        """
        if self.task != self.user_config.default_text:
            cmd = f'bcipy -u "{self.user}" -t "{self.task}" -p "{self.parameter_location}"'
        else:
            cmd = f'bcipy -u "{self.user}" -e "{self.experiment}" -p "{self.parameter_location}"'

        if self.alert:
            cmd += ' -a'

        return cmd

    def _execute_experiment(self, cmd: str) -> None:
        """Execute the experiment command.

        Args:
            cmd (str): The command to execute.
        """
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
        self.button_timeout = 3
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
