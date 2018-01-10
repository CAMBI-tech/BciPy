import unittest

from psychopy import visual
from helpers.load import load_json_parameters
from display.display_main import init_display_window


class TestInitializeDisplayWindow(unittest.TestCase):
    """This is Test Case for Initialzing Display Window."""

    def setUp(self):
        """Set up needed items for test."""

        parameters_used = '../parameters/parameters.json'

        self.parameters = load_json_parameters(parameters_used)

    def test_init_display_window_returns_psychopy_window(self):
        """Test display window is created."""

        test_display_window = visual.Window(
            size=[self.parameters['window_width']['value'],
                  self.parameters['window_height']['value']],
            fullscr=False, screen=0,
            allowGUI=False, allowStencil=False, monitor='mainMonitor',
            color='black', colorSpace='rgb', blendMode='avg',
            waitBlanking=True)

        display_window = init_display_window(self.parameters)

        self.assertEqual(type(display_window), type(test_display_window))
