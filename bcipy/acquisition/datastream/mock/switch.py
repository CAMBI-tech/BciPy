"""Functionality to simulate a switch"""
import logging
import sys

from pylsl import StreamInfo, StreamOutlet

from bcipy.acquisition.devices import (IRREGULAR_RATE, DeviceSpec,
                                       preconfigured_device)
from bcipy.config import SESSION_LOG_FILENAME
from bcipy.gui.main import BCIGui, app

log = logging.getLogger(SESSION_LOG_FILENAME)


def switch_device() -> DeviceSpec:
    """Mock DeviceSpec for a switch.

    Returns:
        DeviceSpec: A DeviceSpec object configured for a switch.
    """
    device = preconfigured_device('Switch', strict=False)
    if device:
        return device
    return DeviceSpec(name='Switch',
                      channels=['Marker'],
                      sample_rate=IRREGULAR_RATE,
                      content_type='Markers')


class Switch:
    """Mock switch which streams data over LSL at an irregular interval."""

    def __init__(self):
        """Initializes the Switch with a DeviceSpec and LSL StreamOutlet."""
        super().__init__()
        self.device: DeviceSpec = switch_device()
        self.lsl_id: str = 'bci_demo_switch'
        info = StreamInfo(self.device.name, self.device.content_type,
                          self.device.channel_count, self.device.sample_rate,
                          self.device.data_type, self.lsl_id)
        self.outlet: StreamOutlet = StreamOutlet(info)

    def click(self, _position: list) -> None:
        """Pushes a sample to the LSL stream when a click event occurs.

        Args:
            _position (list): The position of the click event. This parameter is
                              currently unused.
        """
        log.debug("Click!")
        self.outlet.push_sample([1.0])

    def quit(self) -> None:
        """Cleans up and releases the LSL StreamOutlet."""
        del self.outlet
        self.outlet = None


class SwitchGui(BCIGui):  # pragma: no cover
    """GUI to emulate a switch."""

    def __init__(self, switch: 'Switch', *args, **kwargs):
        """Initializes the SwitchGui with a Switch object.

        Args:
            switch (Switch): The Switch object to control.
            *args: Variable length argument list for the base class.
            **kwargs: Arbitrary keyword arguments for the base class.
        """
        super().__init__(*args, **kwargs)
        self.switch: Switch = switch

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


def main(switch: Switch) -> None:  # pragma: no cover
    """Creates a PyQt6 GUI with a single button in the middle and runs it.

    Performs the switch action when the button is clicked and cleans up
    the switch resources upon exit.

    Args:
        switch (Switch): The Switch object to associate with the GUI.
    """
    gui = app(sys.argv)
    ex = SwitchGui(switch=switch,
                   title='Demo Switch!',
                   height=200,
                   width=300,
                   background_color='black')

    ex.show_gui()
    result = gui.exec()
    switch.quit()
    sys.exit(result)


if __name__ == '__main__':
    main(Switch())
