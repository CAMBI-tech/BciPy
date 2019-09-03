import unittest

from mockito import any, mock, when, unstub
import psychopy
from bcipy.helpers.load import load_json_parameters, PARAM_LOCATION_DEFAULT
from bcipy.display.display_main import init_display_window


class TestInitializeDisplayWindow(unittest.TestCase):
    """This is Test Case for Initialzing Display Window."""

    def setUp(self):
        """Set up needed items for test."""

        parameters_used = PARAM_LOCATION_DEFAULT
        self.window = mock()
        when(psychopy.visual).Window(
            size=any(), screen=any(),
            allowGUI=False,
            useFBO=False,
            fullscr=any(bool),
            allowStencil=False,
            monitor=any(str),
            winType='pyglet',
            units=any(),
            waitBlanking=False,
            color=any(str)).thenReturn(self.window)

        self.parameters = load_json_parameters(parameters_used,
                                               value_cast=True)

        self.display_window = init_display_window(self.parameters)

    def tearDown(self):
        self.display_window.close()
        unstub()

    def test_init_display_window_returns_psychopy_window(self):
        """Test display window is created."""
        self.assertIsInstance(self.display_window, type(self.window))
        self.assertEqual(self.display_window, self.window)
