import subprocess
from bcipy.gui.gui_main import BCIGui
import wx


class RSVPKeyboard(BCIGui):

    event_started = False

    def bind_action(self, action: str, btn: wx.Button) -> None:
        if action == 'launch_bci':
            self.Bind(wx.EVT_BUTTON, self.launch_bci_main, btn)
        elif action == 'edit_parameters':
            self.Bind(wx.EVT_BUTTON, self.edit_parameters, btn)
        elif action == 'refresh':
            self.Bind(wx.EVT_BUTTON, self.refresh, btn)
        elif action == 'offline_analysis':
            self.Bind(wx.EVT_BUTTON, self.offline_analysis, btn)
        else:
            self.Bind(wx.EVT_BUTTON, self.on_clicked, btn)

    def edit_parameters(self, event) -> None:
        """Edit Parameters.

        Function for executing the edit parameter window
        """
        subprocess.call('python bcipy/gui/params_form.py', shell=True)

    def launch_bci_main(self, event: wx.Event) -> None:
        """Launch BCI MAIN"""
        if self.check_input():
            username = self.input_text[0].GetValue().replace(" ", "_")
            experiment_type = self._cast_experiment_type(
                event.GetEventObject().GetLabel())
            mode = 'RSVP'
            cmd = 'python bci_main.py -m {} -t {} -u {}'.format(
                mode, experiment_type, username)

            subprocess.call(cmd, shell=True)
            self.event_started = True

    def check_input(self) -> bool:
        """Check Input."""
        try:
            if self.input_text[0].GetValue() == '':
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

    def offline_analysis(self, event: wx.Event) -> None:
        cmd = 'python bcipy/signal_model/offline_analysis.py'
        subprocess.call(cmd, shell=True)
        event_started = True

    def refresh(self, event: wx.Event) -> None:
        self.event_started = False

    def _cast_experiment_type(self, experiment_type_string: str) -> None:
        if experiment_type_string == 'Calibration':
            experiment_type = 1
        elif experiment_type_string == 'Copy Phrase':
            experiment_type = 2
        elif experiment_type_string == 'Copy Phrase C.':
            experiment_type = 3
        else:
            dialog = wx.MessageDialog(
                self, "Not a registered experiment type!", 'Info',
                wx.OK | wx.ICON_WARNING)
            dialog.ShowModal()
            dialog.Destroy()
            raise ValueError('Register this experiment type or remove button')

        return experiment_type


# Start the app and init the main GUI
app = wx.App(False)
gui = RSVPKeyboard(
    title="RSVPKeyboard", size=(650, 550), background_color='black')

# STATIC TEXT!
gui.add_static_text(
    text='RSVPKeyboard', position=(185, 0), size=25, color='white')
gui.add_static_text(
    text='1.) Enter a User ID:', position=(75, 110), size=15, color='white')
gui.add_static_text(
    text='2.) Chose your experiment type:',
    position=(75, 250), size=15, color='white')

# BUTTONS!
gui.add_button(
    message="Calibration",
    position=(75, 300), size=(100, 100),
    color='grey',
    action='launch_bci')
gui.add_button(
    message="Copy Phrase", position=(200, 300),
    size=(100, 100),
    color='grey',
    action='launch_bci')
gui.add_button(
    message="Copy Phrase C.", position=(325, 300),
    size=(100, 100),
    color='grey',
    action='launch_bci')
gui.add_button(
    message="Free Spell", position=(450, 300),
    size=(100, 100),
    color='grey',
    action='launch_bci')
gui.add_button(
    message='Edit Parameters', position=(0, 450),
    size=(100, 50), color='white',
    action='edit_parameters')
gui.add_button(
    message='Calculate AUC', position=(535, 450),
    size=(100, 50), color='white',
    action='offline_analysis')
gui.add_button(
    message='Refresh', position=(585, 325),
    size=(50, 50), color='white',
    action='refresh')

# TEXT INPUT
gui.add_text_input(position=(75, 150), size=(250, 25))


# IMAGES
gui.add_image(
    path='bcipy/static/images/gui_images/ohsu.png', position=(10, 0), size=100)
gui.add_image(
    path='bcipy/static/images/gui_images/neu.png', position=(530, 0), size=100)


# Make the GUI Show now
gui.show_gui()
app.MainLoop()
