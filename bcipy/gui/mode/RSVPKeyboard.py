import subprocess
from bcipy.gui.gui_main import BCIGui
import wx


class RSVPKeyboard(BCIGui):
    def bind_action(self, action: str, btn: wx.Button) -> None:
        if action == 'launch_bci':
            self.Bind(wx.EVT_BUTTON, self.launch_bci_main, btn)
        elif action == 'edit_parameters':
            self.Bind(wx.EVT_BUTTON, self.edit_parameters, btn)
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
        except:
            dialog = wx.MessageDialog(
                self, "Error, expected input field for this function",
                'Info', wx.OK | wx.ICON_WARNING)
            dialog.ShowModal()
            dialog.Destroy()
            return False
        return True

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
    text='RSVPKeyboard', position=(120, 0), size=40, color='white')
gui.add_static_text(
    text='User ID:', position=(170, 70), size=15, color='white')
gui.add_static_text(
    text='Chose your experiment type:',
    position=(75, 250), size=15, color='white')

# BUTTONS!
gui.add_button(
    message="Calibration",
    position=(75, 300), size=(100, 100),
    color='red',
    action='launch_bci')
gui.add_button(
    message="Copy Phrase", position=(200, 300),
    size=(100, 100),
    color='blue',
    action='launch_bci')
gui.add_button(
    message="Copy Phrase C.", position=(325, 300),
    size=(100, 100),
    color='green',
    action='launch_bci')
gui.add_button(
    message="Free Spell", position=(450, 300),
    size=(100, 100),
    color='orange',
    action='launch_bci')
gui.add_button(
    message='Edit Parameters', position=(0, 450),
    size=(100, 50), color='white',
    action='edit_parameters')

# TEXT INPUT
gui.add_text_input(position=(170, 100), size=(250, 25))


# IMAGES
gui.add_image(
    path='bcipy/static/images/gui_images/ohsu.png', position=(5, 0), size=125)
gui.add_image(
    path='bcipy/static/images/gui_images/neu.png', position=(510, 0), size=125)


# Make the GUI Show now
gui.show_gui()
app.MainLoop()
