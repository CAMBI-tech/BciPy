import subprocess
import wx
from gui.gui_main import BCIGui


# Make a custom GUI class with the events we want
class BCInterface(BCIGui):
    def bind_action(self, action: str, btn: wx.Button) -> None:
        if action == 'launch_mode':
            self.Bind(wx.EVT_BUTTON, self.launch_mode, btn)
        else:
            self.Bind(wx.EVT_BUTTON, self.on_clicked, btn)

    def launch_mode(self, event) -> None:
        mode_label = event.GetEventObject().GetLabel()
        if mode_label == 'RSVP':
            subprocess.Popen('python gui/mode/RSVPKeyboard.py', shell=True)


app = wx.App(False)
gui = BCInterface(
    title="Brain Computer Interface",
    size=(700, 400), background_color='black')

# STATIC TEXT!
gui.add_static_text(
    text='Brain Computer Interface', position=(150, 0), size=25, color='white')

# BUTTONS!
gui.add_button(
    message="RSVP",
    position=(155, 200), size=(100, 100),
    color='red',
    action='launch_mode')
gui.add_button(
    message="Matrix", position=(280, 200),
    size=(100, 100),
    color='blue',
    action='launch_mode')
gui.add_button(
    message="Shuffle", position=(405, 200),
    size=(100, 100),
    color='green',
    action='launch_mode')

# Images
gui.add_image(
    path='./static/images/gui_images/ohsu.png', position=(5, 0), size=125)
gui.add_image(
    path='./static/images/gui_images/neu.png', position=(550, 0), size=125)


# Make the GUI Show now
gui.show_gui()
app.MainLoop()
