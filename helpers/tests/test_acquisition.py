import shutil
import time
import unittest

from helpers.acquisition_related import init_eeg_acquisition
from helpers.load import load_json_parameters
from helpers.save import init_save_data_structure


class TestAcquisition(unittest.TestCase):

    def setUp(self):
        """set up the needed path for load functions."""
        self.parameters_used = './parameters/parameters.json'
        self.parameters = load_json_parameters(self.parameters_used)
        self.data_save_path = 'data/'
        self.user_information = 'test_user_001'

        self.save = init_save_data_structure(
            self.data_save_path,
            self.user_information,
            self.parameters_used)

    def tearDown(self):
        # clean up by removing the data folder we used for testing
        shutil.rmtree(self.save)

    def test_default_values(self):
        print("Testing init_eeg_acquisition with default values.")

        client, server = init_eeg_acquisition(
            self.parameters, self.save, server=True)
        client.start_acquisition()
        time.sleep(0.1)
        client.stop_acquisition()
        client.cleanup()
        server.stop()
        server.join()

        self.assertTrue('raw_data.csv' in client._processor._filename)
        self.assertEqual(client.device_info.name, 'DSI')
        self.assertEqual(client.device_info.fs, 300)

    def test_allows_customization(self):
        print("Testing init_eeg_acquisition with custom values.")

        f = 'foo.csv'
        params = self.parameters
        params['raw_data_name'] = {"value": f}
        params['acq_port'] = {"value": 9000}

        client, server = init_eeg_acquisition(params, self.save, server=True)

        with client:
            time.sleep(0.1)
        client.cleanup()
        server.stop()
        server.join()

        self.assertTrue(f in client._processor._filename)
        self.assertEqual(client.device_info.name, 'DSI')
        self.assertEqual(client.device_info.fs, 300)

    def test_can_use_lsl(self):
        print("Testing init_eeg_acquisition with LSL device")

        params = self.parameters
        params['acq_device'] = {"value": 'LSL'}

        client, server = init_eeg_acquisition(params, self.save, server=True)

        with client:
            time.sleep(0.1)
        client.cleanup()
        server.stop()
        server.join()

        self.assertEqual(client.device_info.name, 'LSL')
        self.assertEqual(client.device_info.fs, 512)


if __name__ == '__main__':
    unittest.main()
