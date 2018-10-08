# pylint: disable=no-member,fixme
"""GUI for running RSVP Tasks"""
import itertools
import os
import subprocess

import wx

from bcipy.gui.gui_main import BCIGui
from bcipy.helpers.load import load_json_parameters

# Order must be the same as the start_task.py experiment types.
# TODO: Encapsulate this logic elsewhere so it can be used in both places.
# TODO: Get tasks by introspecting on Task subclasses (recursively)
TASKS = ['Calibration', 'Copy Phrase', 'Copy Phrase C.', 'Icon to Icon',
         'Icon to Word', 'Alert Tone']


class RSVPKeyboard(BCIGui):
    """GUI for launching the RSVP Tasks."""
    event_started = False
    PARAMETER_LOCATION = 'bcipy/parameters/parameters.json'

    def bind_action(self, action: str, btn) -> None:
        if action == 'launch_bci':
            self.Bind(wx.EVT_BUTTON, self.launch_bci_main, btn)
        elif action == 'edit_parameters':
            self.Bind(wx.EVT_BUTTON, self.edit_parameters, btn)
        elif action == 'refresh':
            self.Bind(wx.EVT_BUTTON, self.refresh, btn)
        elif action == 'offline_analysis':
            self.Bind(wx.EVT_BUTTON, self.offline_analysis, btn)
        elif action == 'load_items_from_txt':
            self.Bind(wx.EVT_COMBOBOX_DROPDOWN, self.load_items_from_txt, btn)
        else:
            self.Bind(wx.EVT_BUTTON, self.on_clicked, btn)

    def edit_parameters(self, _event) -> None:
        """Edit Parameters.

        Function for executing the edit parameter window
        """
        subprocess.call('python bcipy/gui/params_form.py', shell=True)

    def launch_bci_main(self, event: wx.Event) -> None:
        """Launch BCI MAIN"""
        if self.check_input():
            self.event_started = True
            username = self.comboboxes[0].GetValue().replace(" ", "_")
            experiment_type = event.GetEventObject().GetId()
            mode = 'RSVP'
            cmd = 'python bci_main.py -m {} -t {} -u {}'.format(
                mode, experiment_type, username)

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

    def offline_analysis(self, _event: wx.Event) -> None:
        """Run the offline analysis in a new Process."""
        cmd = 'python bcipy/signal/model/offline_analysis.py'
        subprocess.Popen(cmd, shell=True)
        self.event_started = True

    def refresh(self, _event: wx.Event) -> None:
        self.event_started = False

    def load_items_from_txt(self, _event):
        """Loads user directory names from the data path defined in
        parameters.json, and adds those directory names as items to the user id
        selection combobox."""
        parameters = load_json_parameters(
            self.PARAMETER_LOCATION, value_cast=True)
        data_save_loc = parameters['data_save_loc']
        if os.path.isdir(data_save_loc):
            saved_users = os.listdir(data_save_loc)
        elif os.path.isdir('bcipy/' + data_save_loc):
            saved_users = os.listdir('bcipy/' + data_save_loc)
        else:
            raise IOError('User data save location not found')
        self.comboboxes[0].Clear()
        self.comboboxes[0].AppendItems(saved_users)


def main():
    """Create the GUI and run"""
    task_colors = itertools.cycle(
        [wx.Colour(221, 37, 56), wx.Colour(239, 146, 40),
         wx.Colour(239, 212, 105), wx.Colour(117, 173, 48),
         wx.Colour(62, 161, 232), wx.Colour(192, 122, 224),
         wx.Colour(5, 62, 221)])

    side_padding = 15
    btn_width_apart = 105
    btn_padding = 20

    window_width = (len(TASKS) * btn_width_apart) + btn_padding
    # Start the app and init the main GUI
    app = wx.App(False)
    gui = RSVPKeyboard(
        title="RSVPKeyboard", size=(window_width, 550), background_color='black')

    # STATIC TEXT!
    gui.add_static_text(
        text='RSVPKeyboard', position=(185, 0), size=25, color='white')
    gui.add_static_text(
        text='1.) Enter a User ID:', position=(75, 110), size=15, color='white')
    gui.add_static_text(
        text='2.) Choose your experiment type:',
        position=(75, 250), size=15, color='white')

    # BUTTONS!
    btn_size = (85, 80)
    btn_pos_x = side_padding
    btn_pos_y = 300
    for btn_id, btn_label in enumerate(TASKS):
        # The btn_id should correspond to the task_type['exp_type'] value in
        # start_task.py
        gui.add_button(
            message=btn_label,
            position=(btn_pos_x, btn_pos_y),
            size=btn_size,
            color=next(task_colors),
            action='launch_bci',
            id=btn_id + 1)
        btn_pos_x += btn_width_apart

    gui.add_button(
        message='Edit Parameters', position=(side_padding, 450),
        size=(100, 50), color='white',
        action='edit_parameters')

    btn_auc_width = 100
    btn_auc_x = window_width - (2 * side_padding + btn_auc_width)
    gui.add_button(
        message='Calculate AUC', position=(btn_auc_x, 450),
        size=(btn_auc_width, 50), color='white',
        action='offline_analysis')

    btn_refresh_width = 50
    btn_refresh_x = window_width - (2 * side_padding + btn_refresh_width)
    gui.add_button(
        message='Refresh', position=(btn_refresh_x, 230),
        size=(btn_refresh_width, 50), color='white',
        action='refresh')

    # TEXT INPUT
    gui.add_combobox(position=(75, 150), size=(
        250, 25), action='load_items_from_txt')

    # IMAGES
    gui.add_image(
        path='bcipy/static/images/gui_images/ohsu.png', position=(side_padding, 0), size=100)
    gui.add_image(
        path='bcipy/static/images/gui_images/neu.png', position=(530, 0), size=100)

    # Make the GUI Show now
    gui.show_gui()
    app.MainLoop()


if __name__ == "__main__":
    main()
