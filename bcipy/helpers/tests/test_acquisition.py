"""Tests for acquisition helper."""
import shutil
import time
import unittest

from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.helpers.acquisition import init_eeg_acquisition, max_inquiry_duration
from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.save import init_save_data_structure


class TestAcquisition(unittest.TestCase):
    """Unit tests for acquisition helper"""

    def setUp(self):
        """set up the needed path for load functions."""
        self.parameters_used = DEFAULT_PARAMETERS_PATH
        self.parameters = load_json_parameters(self.parameters_used,
                                               value_cast=True)
        self.data_save_path = 'data/'
        self.user_information = 'test_user_001'
        self.task = 'RSVP Calibration'

        self.save = init_save_data_structure(self.data_save_path,
                                             self.user_information,
                                             self.parameters_used, self.task)

    def tearDown(self):
        """Override; teardown test"""
        shutil.rmtree(self.save)

    def test_default_values(self):
        """Test default values."""
        self.parameters['acq_device'] = 'DSI'

        client, server = init_eeg_acquisition(self.parameters,
                                              self.save,
                                              server=True)

        client.start_acquisition()
        time.sleep(0.1)
        client.stop_acquisition()
        client.cleanup()
        server.stop()

        self.assertEqual(client.device_spec.name,
                         self.parameters['acq_device'])
        self.assertEqual(client.device_spec.sample_rate, 300)

    def test_lsl_client(self):
        """Test init_eeg_acquisition with LSL client."""

        params = self.parameters
        params['acq_device'] = 'DSI-24'

        client, server = init_eeg_acquisition(params, self.save, server=True)

        with client:
            time.sleep(0.1)
        client.cleanup()
        server.stop()

        self.assertEqual(client.device_spec.name, 'DSI-24')
        self.assertEqual(client.device_spec.sample_rate, 300)

    def test_max_inquiry_duration(self):
        """Test the max inquiry duration function"""
        params = {
            'time_fixation': 0.5,
            'time_prompt': 1,
            'stim_length': 10,
            'time_flash': 0.25,
            'task_buffer_length': 0.75,
            'prestim_length': 0,
            'stim_jitter': 0
        }

        self.assertEqual(4.75, max_inquiry_duration(params))


if __name__ == '__main__':
    unittest.main()
