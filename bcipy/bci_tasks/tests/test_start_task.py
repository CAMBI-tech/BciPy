import unittest
import shutil
import psychopy

from mockito import any, mock, unstub, when

from helpers.load import load_json_parameters
from helpers.save import init_save_data_structure
from bci_tasks.start_task import start_task


class TestStartTask(unittest.TestCase):
    ''' This is a Test Case for Starting a BCI Task'''

    def setUp(self):
        # set up the needed data to start a task
        parameters_used = '../bci/parameters/parameters.json'
        self.parameters = load_json_parameters(
            parameters_used, value_cast=True)
        self.parameters['num_sti'] = 1

        # Mock the display window
        self.display_window = mock()
        self.display_window.size = [1]

        # Mock the frame rate return
        when(self.display_window).getActualFrameRate().thenReturn(60)
        self.text_stim = mock()
        self.text_stim.height = 2
        self.text_stim.boundingBox = [1]

        # Mock the psychopy text stims and image stims we would expect
        when(psychopy.visual).TextStim(
            win=self.display_window,
            text=any(), font=any()).thenReturn(self.text_stim)
        when(psychopy.visual).TextStim(
            win=self.display_window, color=any(str), height=any(),
            text=any(), font=any(), pos=any(), wrapWidth=any(), colorSpace=any(),
            opacity=any(), depth=any()).thenReturn(self.text_stim)
        when(psychopy.visual).ImageStim(
            self.display_window, image=any(str),
            size=any(), pos=any(), mask=None, ori=any()).thenReturn(self.text_stim)

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
        # clean up by removing the data folder we used for testing
        shutil.rmtree(self.data_save_path)
        unstub()

    def test_start_task_returns_helpful_message_on_undefiend_task(self):
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
