# pylint: disable=no-member
"""GUI for running RSVP tasks"""
import itertools
import os
import subprocess
import sys

from typing import List

from bcipy.gui.gui_main import BCIGui, app, AlertMessageType, AlertResponse
from bcipy.helpers.load import load_json_parameters, copy_parameters
from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH
from bcipy.tasks.task_registry import ExperimentType


class RSVPKeyboard(BCIGui):
    """GUI for launching the RSVP tasks and editing parameters."""

    tasks = ExperimentType.by_mode()['RSVP']

    default_text = '...'
    padding = 15
    btn_width_apart = 105
    btn_padding = 20

    def __init__(self, *args, **kwargs):
        super(RSVPKeyboard, self).__init__(*args, **kwargs)
        self.event_started = False
        self.parameter_location = DEFAULT_PARAMETERS_PATH

        self.parameters = load_json_parameters(self.parameter_location,
                                               value_cast=True)
        self.task_colors = itertools.cycle(
            ['Tomato', 'orange', 'MediumTurquoise', 'MediumSeaGreen', 'MediumPurple', 'Moccasin'])

        # This is set in the build_inputs method that is called automatically when using show_gui
        self.user_input = None

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
                    message=
                    'The selected parameters file is out of date. Would you like to update it with the latest options?',
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
            f'python bcipy/gui/params_form.py -p {self.parameter_location}',
            shell=True)

    def launch_bci_main(self) -> None:
        """Launch BCI main.

        If a user id has been selected, get the id set on the task button and execute
            the task in a new shell.
        """
        if self.check_input():
            self.event_started = True
            username = self.user_input.currentText()
            experiment_type = self.sender().get_id()
            mode = 'RSVP'
            cmd = 'python bci_main.py -m {} -t {} -u {} -p {}'.format(
                mode, experiment_type, username, self.parameter_location)

            subprocess.Popen(cmd, shell=True)

    def check_input(self) -> bool:
        """Check Input.

        Checks to make sure user has input all required fields. Currently, only user id is required.
        """
        try:
            if self.user_input.currentText() == RSVPKeyboard.default_text:
                self.throw_alert_message(
                    title='BciPy Alert',
                    message='Please input a User ID',
                    message_type=AlertMessageType.INFO,
                    okay_to_exit=True)
                return False
            if self.event_started:
                return False
        except Exception as e:
            self.throw_alert_message(
                title='BciPy Alert',
                message=f'Error, {e}',
                message_type=AlertMessageType.CRIT,
                okay_to_exit=True)
            return False
        return True

    def offline_analysis(self) -> None:
        """Run the offline analysis in a new Process."""
        cmd = 'python bcipy/signal/model/offline_analysis.py'
        self.event_started = True
        subprocess.Popen(cmd, shell=True)

    def refresh(self) -> None:
        """Refresh.

        Reset event_started to False to allow a new task to be selected.
        """
        self.event_started = False

    def load_items_from_txt(self) -> List[str]:
        """Load Items From Text.

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

    def fast_scandir(self, directory_name: str, return_path: bool = True) -> List[str]:
        """Fast Scan Directory.

        directory_name: name of the directory to be scanned
        return_path: whether or not to return the scanned directories as a relative path or name.
            False will return the directory name only.
        """
        if return_path:
            return [f.path for f in os.scandir(directory_name) if f.is_dir()]

        return [f.name for f in os.scandir(directory_name) if f.is_dir()]

    def build_buttons(self) -> None:
        """Build buttons.

        Construct all buttons needed for RSVPKeyboard.
        """
        btn_size = [85, 80]
        btn_pos_x = self.padding
        btn_pos_y = 300

        for task in RSVPKeyboard.tasks:
            self.add_button(
                message=task.label,
                position=[btn_pos_x, btn_pos_y],
                size=btn_size,
                background_color=next(self.task_colors),
                action=self.launch_bci_main,
                id=task.value)
            btn_pos_x += self.btn_width_apart

        command_btn_height = 40
        self.add_button(
            message='Load Parameters', position=[self.padding, 450],
            size=[105, command_btn_height], background_color='white',
            action=self.select_parameters)

        self.add_button(
            message='Edit', position=[self.padding + 110, 450],
            size=[50, command_btn_height], background_color='white',
            action=self.edit_parameters)

        btn_auc_width = 100
        btn_auc_x = self.width - (2 * self.padding + btn_auc_width)
        self.add_button(
            message='Calculate AUC', position=(btn_auc_x, 450),
            size=(btn_auc_width, command_btn_height), background_color='white',
            action=self.offline_analysis)

        btn_refresh_width = 50
        btn_refresh_x = self.width - (2 * self.padding + btn_refresh_width)
        self.add_button(
            message='Refresh', position=[btn_refresh_x, 230],
            size=[btn_refresh_width, command_btn_height], background_color='white',
            action=self.refresh)

    def update_user_list(self) -> None:
        """Updates the user_input combo box with a list of user ids based on the
        data directory configured in the current parameters."""

        self.user_input.clear()
        self.user_input.addItem(RSVPKeyboard.default_text)
        self.user_input.addItems(self.load_items_from_txt())

    def build_inputs(self) -> None:
        """Build Inputs.

        Build all inputs needed for RSVPKeyboard.
        """
        self.user_input = self.add_combobox(
            position=[75, 150],
            size=[280, 40],
            items=[],
            editable=True,
            background_color='white',
            text_color='black')

        self.update_user_list()

    def build_images(self) -> None:
        """Build Images.

        Add all images needed for the RSVPKeyboard GUI.
        """
        self.add_image(
            path='bcipy/static/images/gui_images/ohsu.png', position=[self.padding, 0], size=100)
        self.add_image(
            path='bcipy/static/images/gui_images/neu.png', position=[self.width - self.padding - 100, 0], size=100)

    def build_text(self) -> None:
        """Build Text.

        Add all static text needed for RSVPKeyboard. Note: these are textboxes and require that you
            shape the size of the box as well as the text size.
        """
        self.add_static_textbox(
            text='RSVPKeyboard',
            position=[275, 0],
            size=[200, 50],
            background_color='black',
            text_color='white',
            font_size=30)
        self.add_static_textbox(
            text='1.) Enter a User ID:',
            position=[75, 110],
            size=[200, 50],
            background_color='black',
            text_color='white',
            font_size=14)
        self.add_static_textbox(
            text='2.) Choose your experiment type:',
            position=[75, 250],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=14)


def run_rsvp_gui() -> None:
    """Create the GUI and run"""
    bcipy_gui = app(sys.argv)
    window_width = (len(RSVPKeyboard.tasks) * RSVPKeyboard.btn_width_apart) + RSVPKeyboard.btn_padding
    # Start the app and init the main GUI
    gui = RSVPKeyboard(
        title="RSVPKeyboard",
        width=window_width,
        height=550,
        background_color='black')

    # Make the GUI display
    gui.show_gui()

    sys.exit(bcipy_gui.exec_())


if __name__ == "__main__":
    run_rsvp_gui()
