import time
import unittest

from bci.helpers.acquisition import init_acquisition
from mock import mock_open, patch


class TestAcquisition(unittest.TestCase):

    def test_default_values(self):
        params = {}

        m = mock_open()
        with patch('bci.acquisition.processor.open', m):
            client = init_acquisition(params)
            time.sleep(0.1)
            client.stop_acquisition()
            m.assert_called_once_with('rawdata.csv', 'w')
            handle = m()
            assert 'daq_type,DSI' in str(handle.write.mock_calls[0])
            assert 'sample_rate,300' in str(handle.write.mock_calls[1])

    def test_allows_customization(self):
        f = 'foo.csv'
        params = {'connection_params': {'port': 9999}, 'filename': f}

        m = mock_open()
        with patch('bci.acquisition.processor.open', m):
            client = init_acquisition(params)
            time.sleep(0.1)
            client.stop_acquisition()
            m.assert_called_once_with(f, 'w')
            handle = m()
            assert 'daq_type,DSI' in str(handle.write.mock_calls[0])
            assert 'sample_rate,300' in str(handle.write.mock_calls[1])
