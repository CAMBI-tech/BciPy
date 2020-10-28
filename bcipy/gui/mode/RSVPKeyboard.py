# pylint: disable=no-member
"""GUI for running RSVP tasks"""
import datetime
import itertools
import os
import subprocess
import sys
import wx

from bcipy.gui.gui_main import BCIGui, app, PushButton
from bcipy.helpers.load import load_json_parameters, copy_parameters
from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH

from bcipy.tasks.task_registry import ExperimentType

tasks = ExperimentType.by_mode()['RSVP']
# TODO load from config?
side_padding = 15
btn_width_apart = 105
btn_padding = 20

class RSVPKeyboard(BCIGui):
    """GUI for launching the RSVP tasks."""

    def __init__(self, *args, **kwargs):
        super(RSVPKeyboard, self).__init__(*args, **kwargs)
        self.event_started = False
        self.parameter_location = DEFAULT_PARAMETERS_PATH

        self.task_colors = itertools.cycle(
            ['blue', 'green', 'yellow', 'red', 'limegreen', 'gray', 'pink'])

    def select_parameters(self) -> None:
        """Dialog to select the parameters.json configuration to use."""
        with wx.FileDialog(self,
                           'Select parameters file',
                           wildcard='JSON files (*.json)|*.json',
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fd:
            if fd.ShowModal() != wx.ID_CANCEL:
                self.parameter_location = fd.GetPath()

    def edit_parameters(self) -> None:
        """Edit Parameters. Prompts for a parameters.json file to use. If the default parameters
        are selected a copy is used.
        """
        if self.parameter_location == DEFAULT_PARAMETERS_PATH:
            # Don't allow the user to overwrite the defaults
            with wx.MessageDialog(
                    self,
                    "The default parameters.json can't be overridden. A copy will be used.",
                    'Info', wx.OK | wx.CANCEL) as confirm_dialog:
                response = confirm_dialog.ShowModal()
                if response == wx.ID_OK:
                    self.parameter_location = copy_parameters()
                else:
                    return

        subprocess.call(
            f'python bcipy/gui/params_form.py -p {self.parameter_location}',
            shell=True)

    def launch_bci_main(self) -> None:
        """Launch BCI MAIN"""
        if self.check_input():
            self.event_started = True
            username = self.comboboxes[0].GetValue().replace(" ", "_")
            experiment_type = event.GetEventObject().GetId()
            mode = 'RSVP'
            cmd = 'python bci_main.py -m {} -t {} -u {} -p {}'.format(
                mode, experiment_type, username, self.parameter_location)

            subprocess.Popen(cmd, shell=True)

    def check_input(self) -> bool:
        """Check Input."""
        try:
            if self.comboboxes[0].GetValue() == '':
                dialog = wx.MessageDialog(
                    self, "Please Input User ID", 'Info',
                    wx.OK | wx.ICON_WARNING)
                dialog.ShowModal()
                dialog.Destroy()
                return False
            if self.event_started:
                return False
        except Exception as e:
            dialog = wx.MessageDialog(
                self, f'Error, {e}',
                'Info', wx.OK | wx.ICON_WARNING)
            dialog.ShowModal()
            dialog.Destroy()
            return False
        return True

    def offline_analysis(self) -> None:
        """Run the offline analysis in a new Process."""
        cmd = 'python bcipy/signal/model/offline_analysis.py'
        self.event_started = True
        subprocess.Popen(cmd, shell=True)

    def refresh(self) -> None:
        self.event_started = False

    def load_items_from_txt(self):
        """Loads user directory names from the data path defined in
        parameters.json, and adds those directory names as items to the user id
        selection combobox."""
        saved_users = None
        parameters = load_json_parameters(
            self.parameter_location, value_cast=True)
        data_save_loc = parameters['data_save_loc']
        if os.path.isdir(data_save_loc):
            saved_users = os.listdir(data_save_loc)
        elif os.path.isdir('bcipy/' + data_save_loc):
            saved_users = os.listdir('bcipy/' + data_save_loc)
        else:
            raise IOError('User data save location not found')
        return saved_users

    def build_buttons(self):
        btn_size = [85, 80]
        btn_pos_x = side_padding
        btn_pos_y = 300
        for task in tasks:
            self.add_button(
                message=task.label,
                position=[btn_pos_x, btn_pos_y],
                size=btn_size,
                background_color=next(self.task_colors),
                action=self.launch_bci_main,
                id=task.value)
            btn_pos_x += btn_width_apart

        command_btn_height = 40
        self.add_button(
            message='Load Parameters', position=[side_padding, 450],
            size=[105, command_btn_height], background_color='white',
            action=self.select_parameters)
        self.add_button(
            message='Edit', position=[side_padding + 110, 450],
            size=[50, command_btn_height], background_color='white',
            action=self.edit_parameters)

        btn_auc_width = 100
        btn_auc_x = self.width - (2 * side_padding + btn_auc_width)
        self.add_button(
            message='Calculate AUC', position=(btn_auc_x, 450),
            size=(btn_auc_width, command_btn_height), background_color='white',
            action=self.offline_analysis)

        btn_refresh_width = 50
        btn_refresh_x = self.width - (2 * side_padding + btn_refresh_width)
        self.add_button(
            message='Refresh', position=[btn_refresh_x, 230],
            size=[btn_refresh_width, command_btn_height], background_color='white',
            action=self.refresh)

    def build_inputs(self):
        items = self.load_items_from_txt()
        self.add_combobox(position=[75, 150], size=[250, 25], items=items)

    def build_images(self):
        self.add_image(
            path='bcipy/static/images/gui_images/ohsu.png', position=[side_padding, 0], size=100)
        self.add_image(
            path='bcipy/static/images/gui_images/neu.png', position=[530, 0], size=100)

    def build_text(self):
        self.add_static_text(
            text='RSVPKeyboard', position=[185, 0], size=[25, 25], text_color='white')
        self.add_static_text(
            text='1.) Enter a User ID:', position=[75, 110], size=[15, 15], text_color='white')
        self.add_static_text(
            text='2.) Choose your experiment type:',
            position=[75, 250], size=[15, 15], text_color='white')

    def build_assets(self):

        self.build_buttons()
        self.build_images()
        self.build_text()


def run_rsvp_gui():
    """Create the GUI and run"""
    bcipy_gui = app(sys.argv)
    window_width = (len(tasks) * btn_width_apart) + btn_padding
    # Start the app and init the main GUI
    gui = RSVPKeyboard(
        title="RSVPKeyboard",
        width=window_width,
        height=550,
        background_color='black')


    # Make the GUI Show now
    gui.show_gui()

    sys.exit(bcipy_gui.exec_())


if __name__ == "__main__":
    run_rsvp_gui()
