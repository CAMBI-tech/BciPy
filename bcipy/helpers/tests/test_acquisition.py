"""Tests for acquisition helper."""
import shutil
import time
import unittest

from bcipy.helpers.acquisition import init_eeg_acquisition
from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.save import init_save_data_structure


class TestAcquisition(unittest.TestCase):
    """Unit tests for acquisition helper"""

    def setUp(self):
        """set up the needed path for load functions."""
        self.parameters_used = 'bcipy/parameters/parameters.json'
        self.parameters = load_json_parameters(self.parameters_used,
                                               value_cast=True)
        self.data_save_path = 'data/'
        self.user_information = 'test_user_001'
        self.task = 'RSVP Calibration'

        self.save = init_save_data_structure(self.data_save_path,
                                             self.user_information,
                                             self.parameters_used,
                                             self.task)

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

        self.assertEqual(client.device_info.name,
                         self.parameters['acq_device'])
        self.assertEqual(client.device_info.fs, 300)

    def test_allows_customization(self):
        """Test init_eeg_acquisition with custom values."""

        params = self.parameters
        params['raw_data_name'] = 'foo.csv'
        params['acq_port'] = 9000
        params['acq_device'] = 'DSI'

        client, server = init_eeg_acquisition(params, self.save, server=True)

        with client:
            time.sleep(0.1)
        client.cleanup()
        server.stop()

        self.assertEqual(client.device_info.name, params['acq_device'])
        self.assertEqual(client.device_info.fs, 300)

    def test_can_use_lsl(self):
        """Test init_eeg_acquisition with LSL device"""

        params = self.parameters
        params['acq_device'] = 'LSL'

        client, server = init_eeg_acquisition(params, self.save, server=True)

        with client:
            time.sleep(0.1)
        client.cleanup()
        server.stop()

        self.assertEqual(client.device_info.name, 'LSL')
        self.assertEqual(client.device_info.fs, 256)


if __name__ == '__main__':
    unittest.main()
