import unittest
import shutil
import psychopy

from mockito import any, mock, unstub, when

from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.save import init_save_data_structure
from bcipy.bci_tasks.start_task import start_task


class TestStartTask(unittest.TestCase):
    """This is a Test Case for Starting a BCI Task."""

    def setUp(self):
        """Set Up."""
        # set up the needed data to start a task
        parameters_used = 'bcipy/parameters/parameters.json'
        self.parameters = load_json_parameters(
            parameters_used, value_cast=True)
        self.parameters['num_sti'] = 1

        # Mock the display window
        self.display_window = mock()
        self.display_window.size = [1, 1]

        # Mock the frame rate return
        when(self.display_window).getActualFrameRate().thenReturn(60)
        self.text_stim = mock()
        self.text_stim.height = 2
        self.text_stim.boundingBox = [1]
        self.text_stim.size = [100.0, 100.0]
        self.text_stim.win = mock()
        self.text_stim.win.size = (500, 500)

        # Mock the psychopy text stims and image stims we would expect
        when(psychopy.visual).TextStim(
            win=self.display_window,
            text=any(), font=any()).thenReturn(self.text_stim)
        when(psychopy.visual).TextStim(
            win=self.display_window, color=any(str), height=any(),
            text=any(), font=any(), pos=any(), wrapWidth=any(), colorSpace=any(),
            opacity=any(), depth=any()).thenReturn(self.text_stim)
        when(psychopy.visual).ImageStim(
            win=self.display_window, image=any(), mask=None,
            pos=any(), ori=any()).thenReturn(self.text_stim)


        # save data information
        self.data_save_path = 'data/'
        self.user_information = 'test_user_001'
        self.file_save = init_save_data_structure(
            self.data_save_path,
            self.user_information,
            parameters_used)
        # Mock the data acquistion
        self.daq = mock()
        self.daq.is_calibrated = True
        self.daq.marker_writer = None

    def tearDown(self):
        """Tear Down."""
        # clean up by removing the data folder we used for testing
        shutil.rmtree(self.data_save_path)
        unstub()

    def test_start_task_raises_exception_on_undefiend_task(self):
        """Exception on undefined mode."""
        task_type = {
            'mode': 'New Mode',
            'exp_type': 1}
        with self.assertRaises(Exception):
            start_task(
                self.display_window,
                self.daq,
                task_type,
                self.parameters,
                self.file_save)

    def test_start_task_runs_rsvp_calibration(self):
        """Start Task RSVP Calibration."""
        when(psychopy.event).getKeys(keyList=any(list)).thenReturn(['space'])
        task_type = {
            'mode': 'RSVP',
            'exp_type': 1
        }
        start_task(
            self.display_window,
            self.daq,
            task_type,
            self.parameters,
            self.file_save)
