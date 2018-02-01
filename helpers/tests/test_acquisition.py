import pytest
import time
import unittest

from helpers.acquisition_related import init_eeg_acquisition
from helpers.load import load_json_parameters
from helpers.save import init_save_data_structure
from mock import mock_open, patch


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

    def test_default_values(self):
        m = mock_open()
        with patch('acquisition.processor.open', m):
            client, server = init_eeg_acquisition(self.parameters, self.save, server=True)
            client.start_acquisition()
            time.sleep(0.1)
            client.stop_acquisition()
            server.stop()
            handle = m()
            assert 'daq_type,DSI' in str(handle.write.mock_calls[0])
            assert 'sample_rate,300' in str(handle.write.mock_calls[1])

    def test_allows_customization(self):
        f = 'foo.csv'
        params = self.parameters
        params['connection_params'] = {'port': 9999}
        params['filename'] = f

        m = mock_open()
        with patch('acquisition.processor.open', m):
            client, server = init_eeg_acquisition(params, self.save, server=True)
            with client:
                time.sleep(0.1)
            server.stop()
            handle = m()
            assert 'daq_type,DSI' in str(handle.write.mock_calls[0])
            assert 'sample_rate,300' in str(handle.write.mock_calls[1])

    def test_accepts_clock(self):
        class _MockClock(object):
            """Clock that acts as a counter."""

            def __init__(self):
                super(_MockClock, self).__init__()
                self.count = 0

            def reset(self):
                self.count = 0

            def getTime(self):
                self.count += 1
                return float(self.count)

        clock = _MockClock()
        m = mock_open()
        with patch('acquisition.processor.open', m):
            client, server = init_eeg_acquisition(self.parameters, self.save,
                                                  clock=clock, server=True)

            with client:
                time.sleep(0.1)

            server.stop()

            data = client.get_data()
            assert clock.count > 0
            assert len(data) == clock.count
