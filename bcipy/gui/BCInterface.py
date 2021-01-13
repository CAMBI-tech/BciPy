import os
import subprocess
import sys

from typing import List

from bcipy.gui.gui_main import BCIGui, app, AlertMessageType, AlertResponse
from bcipy.helpers.load import load_json_parameters, load_experiments, copy_parameters
from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH
from bcipy.tasks.task_registry import TaskType


class BCInterface(BCIGui):
    """BCI Interface.

    Main interface for execution of BciPy experiments and tasks. Additionally, quick access to parameter
        editing and loading, and offline analysis execution.
    """

    tasks = TaskType.list()

    default_text = '...'
    padding = 30
    btn_height = 40

    def __init__(self, *args, **kwargs):
        super(BCInterface, self).__init__(*args, **kwargs)
        self.parameter_location = DEFAULT_PARAMETERS_PATH

        self.parameters = load_json_parameters(self.parameter_location,
                                               value_cast=True)

        # These are set in the build_inputs and represent text inputs from the user
        self.user_input = None
        self.experiment_input = None
        self.task_input = None

        self.user = None
        self.experiment = None
        self.task = None

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
        btn_start_y = 450
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
        subprocess.call(
            f'python bcipy/gui/experiments/ExperimentRegistry.py',
            shell=True)

        self.update_experiment_list()

    def fast_scandir(self, directory_name: str, return_path: bool = True) -> List[str]:
        """Fast Scan Directory.

        directory_name: name of the directory to be scanned
        return_path: whether or not to return the scanned directories as a relative path or name.
            False will return the directory name only.
        """
        if return_path:
            return [f.path for f in os.scandir(directory_name) if f.is_dir()]

        return [f.name for f in os.scandir(directory_name) if f.is_dir()]

    def update_user_list(self) -> None:
        """Updates the user_input combo box with a list of user ids based on the
        data directory configured in the current parameters."""

        self.user_input.clear()
        self.user_input.addItem(BCInterface.default_text)
        self.user_input.addItems(self.load_users())

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
            font_size=24)
        self.add_static_textbox(
            text='Experiment',
            position=[text_x, 205],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=24)
        self.add_static_textbox(
            text='Task',
            position=[text_x, 305],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=24)

    def build_images(self) -> None:
        """Build Images.

        Build add images needed for the UI. In this case, the OHSU and NEU logos.
        """
        self.add_image(
            path='bcipy/static/images/gui_images/ohsu.png', position=[self.padding, 0], size=100)
        self.add_image(
            path='bcipy/static/images/gui_images/neu.png', position=[self.width - self.padding - 110, 0], size=100)

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
        # update GUI options
        if self.user_input:
            self.update_user_list()

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
                    okay_or_cancel=True)

                if save_response == AlertResponse.OK.value:
                    self.parameters.save()

    def edit_parameters(self) -> None:
        """Edit Parameters.

        Prompts for a parameters.json file to use. If the default parameters are selected, a copy is used.
        Note that any edits to the parameters file will not be applied to this GUI until the parameters
        are reloaded.
        """
        if self.parameter_location == DEFAULT_PARAMETERS_PATH:
            # Don't allow the user to overwrite the defaults
            response = self.throw_alert_message(
                title='BciPy Alert',
                message='The default parameters.json cannot be overridden. A copy will be used.',
                message_type=AlertMessageType.INFO,
                okay_or_cancel=True)

            if response == AlertResponse.OK.value:
                self.parameter_location = copy_parameters()
            else:
                return None

        subprocess.call(
            f'python bcipy/gui/parameters/params_form.py -p {self.parameter_location}',
            shell=True)

    def check_input(self) -> bool:
        """Check Input.

        Checks to make sure user has input all required fields. Currently, only user id is required.
        """

        # Update based on current inputs
        self.user = self.user_input.currentText()
        self.experiment = self.experiment_input.currentText()
        self.task = self.task_input.currentText()

        # Check the set values are different than defaults
        try:
            if self.user == BCInterface.default_text:
                self.throw_alert_message(
                    title='BciPy Alert',
                    message='Please input a User ID',
                    message_type=AlertMessageType.INFO,
                    okay_to_exit=True)
                return False
            if self.experiment == BCInterface.default_text:
                self.throw_alert_message(
                    title='BciPy Alert',
                    message='Please select or create an Experiment',
                    message_type=AlertMessageType.INFO,
                    okay_to_exit=True)
                return False
            if self.task == BCInterface.default_text:
                self.throw_alert_message(
                    title='BciPy Alert',
                    message='Please select a Task',
                    message_type=AlertMessageType.INFO,
                    okay_to_exit=True)
                return False
        except Exception as e:
            self.throw_alert_message(
                title='BciPy Alert',
                message=f'Error, {e}',
                message_type=AlertMessageType.CRIT,
                okay_to_exit=True)
            return False
        return True

    def load_users(self) -> List[str]:
        """Load Users.

        Loads user directory names below experiments from the data path defined in parameters.json
        and returns them as a list.
        """
        saved_users = []
        data_save_loc = self.parameters['data_save_loc']

        # check the directory is valid
        if os.path.isdir(data_save_loc):
            path = data_save_loc

        elif os.path.isdir(f'bcipy/{data_save_loc}'):
            path = f'bcipy/{data_save_loc}'

        else:
            self.logger.info('User data save location not found! Please enter a new user id.')
            return saved_users

        # grab all experiments in the directory and iterate over them to get the users
        experiments = self.fast_scandir(path, return_path=True)

        for experiment in experiments:
            users = self.fast_scandir(experiment, return_path=False)
            saved_users.extend(users)

        return saved_users

    def load_experiments(self) -> List[str]:
        """Load experiments

        Loads experiments registered in the DEFAULT_EXPERIMENT_PATH.
        """
        return load_experiments().keys()

    def start_experiment(self) -> None:
        """Start Experiment Session.

        Using the inputs gathers, check for validity using the check_input method, then launch the experiment using a
            command to bci_main.py and subprocess.
        """
        if self.check_input():
            self.throw_alert_message(
                    title='BciPy Alert',
                    message='Task Starting ...',
                    message_type=AlertMessageType.INFO,
                    okay_to_exit=False)
            cmd = (
                f'python bci_main.py -e "{self.experiment}" '
                f'-u "{self.user}" -t "{self.task}" -p "{self.parameter_location}"'
            )
            subprocess.Popen(cmd, shell=True)
            self.close()

    def offline_analysis(self) -> None:
        """Offline Analysis.

        Run offline analysis as a script in a new process.
        """
        cmd = 'python bcipy/signal/model/offline_analysis.py'
        subprocess.Popen(cmd, shell=True)


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
