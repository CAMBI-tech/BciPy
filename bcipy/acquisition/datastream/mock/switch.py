"""Functionality to simulate a switch"""
import logging
import sys

from pylsl import StreamInfo, StreamOutlet

from bcipy.acquisition.devices import DeviceSpec, IRREGULAR_RATE
from bcipy.gui.main import BCIGui, app

log = logging.getLogger(__name__)


def switch_device() -> DeviceSpec:
    """Mock DeviceSpec for a switch"""
    return DeviceSpec(name='Switch',
                      channels=['Marker'],
                      sample_rate=IRREGULAR_RATE,
                      content_type='Markers')


class Switch:
    """Mock switch which streams data over LSL at an irregular interval."""

    def __init__(self):
        super().__init__()
        self.device = switch_device()
        self.lsl_id = 'bci_demo_switch'
        info = StreamInfo(self.device.name, self.device.content_type,
                          self.device.channel_count, self.device.sample_rate,
                          self.device.data_type, self.lsl_id)
        self.outlet = StreamOutlet(info)

    def click(self, _position):
        """Click event that pushes a sample"""
        log.debug("Click!")
        self.outlet.push_sample([1.0])

    def quit(self):
        """Quit and cleanup"""
        del self.outlet
        self.outlet = None


class SwitchGui(BCIGui):
    """GUI to emulate a switch."""

    def __init__(self, switch: Switch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.switch = switch

    def build_buttons(self) -> None:
        """Build all buttons necessary for the UI.
        """
        self.add_button(message='Click!',
                        position=[100, 75],
                        size=[50, 50],
                        background_color='white',
                        action=self.switch.click)

    def build_text(self) -> None:
        """Build all static text needed for the UI.
        Positions are relative to the height / width of the UI defined in start_app.
        """
        self.add_static_textbox(
            text='Click the button to emulate a switch hit.',
            position=[5, 0],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=16)


def main(switch: Switch):
    """Creates a PyQt5 GUI with a single button in the middle. Performs the
    switch action when clicked."""
    gui = app(sys.argv)
    ex = SwitchGui(switch=switch,
                   title='Demo Switch!',
                   height=200,
                   width=300,
                   background_color='black')

    ex.show_gui()
    result = gui.exec_()
    switch.quit()
    sys.exit(result)


if __name__ == '__main__':
    main(Switch())
