import subprocess
import sys
from bcipy.gui.gui_main import BCIGui, app


class BCInterface(BCIGui):
    """BCInterface. Main GUI to select different modes."""

    def launch_rsvp(self) -> None:
        self.logger.debug('Launching RSVPKeyboard')
        subprocess.Popen(
            'python bcipy/gui/mode/RSVPKeyboard.py',
            shell=True)

    def build_buttons(self) -> None:
        """Build buttons.

        Construct all buttons needed for BCInterface.
        """
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

    def build_images(self) -> None:
        """Build Images.

        Add all images needed for the BCInterface.
        """
        self.add_image(
            path='bcipy/static/images/gui_images/ohsu.png', position=[5, 0], size=100)
        self.add_image(
            path='bcipy/static/images/gui_images/neu.png', position=[self.width - 100, 0], size=100)

    def build_text(self) -> None:
        """Build Text.

        Add all static text needed for RSVPKeyboard. Note: these are textboxes and require that you
            shape the size of the box as well as the text size.
        """
        self.add_static_textbox(
            text='Brain Computer Interface',
            position=[175, 0],
            size=[325, 100],
            background_color='black',
            text_color='white',
            font_size=30)

    def build_assets(self) -> None:
        self.build_buttons()
        self.build_images()
        self.build_text()


def start_app():
    """Start BCIGui."""
    bcipy_gui = app(sys.argv)
    ex = BCInterface(
        title="Brain Computer Interface",
        height=400,
        width=700,
        background_color='black')

    ex.show_gui()

    sys.exit(bcipy_gui.exec_())


if __name__ == '__main__':
    start_app()
