import unittest

from mock import mock_open, patch
from mockito import any, mock, unstub, when
from psychopy.visual import Window

from bcipy.display.display_main import init_display_window
from bcipy.helpers.load import load_json_parameters
from bcipy.tasks.rsvp.icon_to_icon import RSVPIconToIconTask


class TestIconToIcon(unittest.TestCase):
    """Tests for Icon to Icon class"""

    def setUp(self):
        """set up the needed path for load functions."""
        params_path = 'bcipy/parameters/parameters.json'
        self.parameters = load_json_parameters(params_path, value_cast=True)

    def test_img_path(self):
        """Test img_path method"""

        parameters = self.parameters
        parameters['window_height'] = 1
        parameters['window_width'] = 1
        parameters['is_txt_sti'] = False
        img_path = 'bcipy/static/images/rsvp_images/'
        parameters['path_to_presentation_images'] = img_path

        fixation = 'bcipy/static/images/bci_main_images/PLUS.png'
        # TODO: can this be mocked?
        display = init_display_window(parameters)
        daq = mock()
        file_save = ""
        signal_model = None
        language_model = None
        fake = True
        auc_filename = ""

        with patch('bcipy.tasks.rsvp.icon_to_icon.open', mock_open()):
            task = RSVPIconToIconTask(display, daq, parameters, file_save,
                                      signal_model, language_model, fake,
                                      False, auc_filename)
            self.assertTrue(len(task.alp) > 0)
            self.assertTrue('PLUS' not in task.alp)

            self.assertEqual('bcipy/static/images/rsvp_images/A.png',
                             task.img_path('A'))
            self.assertEqual('A.png', task.img_path('A.png'))
            self.assertEqual(fixation, task.img_path(fixation))
        display.close()
