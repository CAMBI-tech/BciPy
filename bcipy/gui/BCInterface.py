import subprocess
import wx
import sys
from bcipy.gui.gui_main import BCIGui, app, PushButton


# Make a custom GUI class with the events we want
class BCInterface(BCIGui):

    def launch_rsvp(self) -> None:
        print('launching')
        subprocess.Popen(
            'python bcipy/gui/mode/RSVPKeyboard.py',
            shell=True)

    def build_buttons(self):
        self.add_button(
            message="RSVP",
            position=[155, 200], size=[100, 100],
            background_color='red',
            action=self.launch_rsvp)
        self.add_button(
            message="Matrix", position=[280, 200],
            size=[100, 100],
            background_color='blue')
        self.add_button(
            message="Shuffle", position=[405, 200],
            size=[100, 100],
            background_color='green')

    def build_images(self):
        self.add_image(
            path='bcipy/static/images/gui_images/ohsu.png', position=[5, 0], size=200)
        self.add_image(
            path='bcipy/static/images/gui_images/neu.png', position=[550, 0], size=200)

    def build_text(self):
        self.add_static_text(text='Brain Computer Interface', position=[250, 0], size=[250, 100], background_color='black', text_color='white')

    def build_assets(self):
        self.build_buttons()
        self.build_images()
        self.build_text()


def start_app():
    """Start BCIGui."""
    bcipy_gui = app(sys.argv)
    ex = BCInterface(
        title="Brain Computer Interface",
        height=500,
        width=700,
        background_color='black')

    ex.show_gui()

    sys.exit(bcipy_gui.exec_())

if __name__ == '__main__':
    start_app()
