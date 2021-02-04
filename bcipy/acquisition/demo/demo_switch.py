"""Functionality to simulate a switch"""
import pygame
import time
import sys
from pylsl import StreamInfo, StreamOutlet
from bcipy.acquisition import devices
from bcipy.gui.gui_main import app, BCIGui
import logging
log = logging.getLogger(__name__)


class Switch:
    """Mock switch which streams data over LSL at an irregular interval."""

    def __init__(self):
        super().__init__()
        self.device = devices.DeviceSpec(name='Switch',
                                         channels=['Btn1'],
                                         sample_rate=devices.IRREGULAR_RATE,
                                         content_type='Markers')
        self.lsl_id = 'bci_demo_switch'
        info = StreamInfo(self.device.name, self.device.content_type,
                          self.device.channel_count, self.device.sample_rate,
                          self.device.data_type, self.lsl_id)
        self.outlet = StreamOutlet(info)

    def click(self, position):
        log.debug("Click!")
        self.outlet.push_sample([1.0])

    def quit(self):
        del self.outlet
        self.outlet = None


class SwitchGui(BCIGui):
    def __init__(self, switch: Switch, *args, **kwargs):
        super(SwitchGui, self).__init__(*args, **kwargs)
        self.switch = switch

    # @override
    def build_buttons(self) -> None:
        """Build all buttons necessary for the UI.
        """
        self.add_button(message='Click!',
                        position=[100, 75],
                        size=[50, 50],
                        background_color='white',
                        action=self.switch.click)

    # @override
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
    # pyqt_gui()
    main(Switch())