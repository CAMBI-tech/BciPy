import unittest
import shutil
from unittest.mock import MagicMock
from helpers.load import load_json_parameters
from helpers.save import init_save_data_structure
from display.display_main import init_display_window
from bci_tasks.start_task import start_task


class TestStartTask(unittest.TestCase):
    ''' This is a Test Case for Starting a BCI Task'''

    def setUp(self):
        # set up the needed data to start a task

        self.task_type = {
            'mode': 'New Mode',
            'exp_type': 1}

        parameters_used = '../bci/parameters/parameters.json'

        self.parameters = load_json_parameters(
            parameters_used, value_cast=True)
        self.display_window = init_display_window(self.parameters)

        self.data_save_path = 'data/'
        self.user_information = 'test_user_001'

        self.file_save = init_save_data_structure(
            self.data_save_path,
            self.user_information,
            parameters_used)

        self.daq = MagicMock()
        self.daq.is_calibrated = True

    def tearDown(self):
        # clean up by removing the data folder we used for testing
        shutil.rmtree(self.data_save_path)

    def test_start_task_returns_helpful_message_on_undefiend_task(self):
        with self.assertRaises(Exception):
            start_task(
                self.display_window,
                self.daq,
                self.task_type,
                self.parameters,
                self.file_save)

    def test_start_task_runs_rsvp_calibration(self):
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
