"""Functions to generate EEG(-like) data for testing and development.
Generators are used by a Producer to stream the data at a given frequency.
"""

import time
import unittest
from bcipy.acquisition.client import DataAcquisitionClient, CountClock
from bcipy.acquisition.processor import Processor, NullProcessor
from bcipy.acquisition.protocols.device import Device
from bcipy.acquisition.util import mock_data, mock_record


class _MockDevice(Device):
    """Device that mocks reading data. Does not need a server. Continues as
    long as there is mock data."""

    def __init__(self, data=None, channels=None, fs=500):
        data = data or []
        channels = channels or []
        super(_MockDevice, self).__init__(
            connection_params={}, channels=channels, fs=fs)
        self.data = data
        self.i = 0

    def name(self):
        return 'MockDevice'

    def read_data(self):
        if self.i < len(self.data):
            row = self.data[self.i]
            self.i += 1
            return row
        return None


class _AccumulatingProcessor(Processor):
    """Processor that records all data passed to the process method."""

    def __init__(self, q):
        super(_AccumulatingProcessor, self).__init__()
        self.data = q

    def process(self, record, timestamp=None):
        self.data.put(record)


class TestDataAcquisitionClient(unittest.TestCase):
    """Main Test class for DataAcquisitionClient code."""

    def __init__(self, *args, **kwargs):
        super(TestDataAcquisitionClient, self).__init__(*args, **kwargs)
        num_channels = 25
        num_records = 500
        self.mock_channels = ['ch' + str(i) for i in range(num_channels)]
        self.mock_data = list(mock_data(num_records, num_channels))

    def test_acquisition_null_device_exception(self):
        """Exception should be thrown if unable to connect to device or message not understood """
        daq = DataAcquisitionClient(device=None)
        daq._is_streaming = False
        with self.assertRaises(Exception):
            daq.start_acquisition()

        daq.cleanup()

    def test_daq_with_no_buffer(self):
        """get_data should return an empty list if daq._buf is None
        data length should return 0
        """

        device = _MockDevice(data=self.mock_data, channels=self.mock_channels)
        daq = DataAcquisitionClient(device=device,
                                    delete_archive=True,
                                    raw_data_file_name=None)
        daq.start_acquisition()
        time.sleep(0.1)
        daq.stop_acquisition()

        # Make sure we are able to stop the buffer process
        buf_temp = daq._buf

        daq._buf = None

        # test get_data
        data = daq.get_data()
        self.assertEqual(data, [])

        # test get_data_len
        data_length = daq.get_data_len()
        self.assertEqual(data_length, 0)

        # test offset
        offset = daq.offset
        self.assertEqual(offset, None)

        daq._buf = buf_temp
        daq.cleanup()

    def test_get_data(self):
        """Data should be queryable."""

        device = _MockDevice(data=self.mock_data, channels=self.mock_channels)
        daq = DataAcquisitionClient(device=device,
                                    delete_archive=True,
                                    raw_data_file_name=None)
        daq.start_acquisition()
        time.sleep(0.1)
        daq.stop_acquisition()

        # Get all records
        data = daq.get_data()
        self.assertTrue(
            len(data) > 0, "DataAcquisitionClient should have data.")
        for i, record in enumerate(data):
            self.assertEqual(record.data, self.mock_data[i])

        daq.cleanup()

    def test_count_clock_get_time(self):
        """getTime should increment count clock """
        clock = CountClock()
        old_counter = clock.counter
        result = clock.getTime()
        self.assertEqual(result, old_counter + 1)

    def test_clock(self):
        """Test clock integration."""

        clock = CountClock()
        clock.counter = 10  # ensures that clock gets reset.

        daq = DataAcquisitionClient(
            device=_MockDevice(data=self.mock_data,
                               channels=self.mock_channels),
            buffer_name='buffer_client_test_clock.db',
            raw_data_file_name=None,
            delete_archive=True,
            clock=clock)
        with daq:
            time.sleep(0.1)

        # Get all records from buffer
        data = daq.get_data()

        # NOTE: we can't make any assertions about the Clock, since it is
        # copied when it's passed to the acquisition thread.
        self.assertTrue(len(data) > 0)
        for i, record in enumerate(data):
            self.assertEqual(record.timestamp, float(i + 1))

        daq.cleanup()

    def test_offset(self):
        """Test offset calculation"""

        channels = ['ch1', 'ch2', 'TRG']
        sample_hz = 100
        trigger_at = 10
        num_records = 500
        n_channels = len(channels) - 1

        data = [mock_record(n_channels) + [0 if (i + 1) < trigger_at else 1]
                for i in range(num_records)]

        device = _MockDevice(data=data, channels=channels, fs=sample_hz)
        daq = DataAcquisitionClient(device=device,
                                    buffer_name='buffer_client_test_offset.db',
                                    raw_data_file_name=None,
                                    delete_archive=True,
                                    clock=CountClock())

        daq.start_acquisition()
        time.sleep(0.1)
        daq.stop_acquisition()

        # The assertions should work before the stop_acquisition, but on some
        # Windows environments the tests were taking too long to setup and the
        # time would complete before any data had been processed.
        self.assertTrue(daq.is_calibrated)
        self.assertEqual(daq.offset, float(trigger_at) / sample_hz)

        daq.cleanup()

    def test_missing_offset(self):
        """Test missing offset when no triggers are present in the data."""
        channels = ['ch1', 'ch2', 'TRG']
        sample_hz = 100
        num_records = 500

        # mock_data only has empty trigger values.
        n_channels = len(channels) - 1
        data = [mock_record(n_channels) + [0] for i in range(num_records)]

        device = _MockDevice(data=data, channels=channels, fs=sample_hz)
        daq = DataAcquisitionClient(
            device=device,
            clock=CountClock(),
            buffer_name='buffer_client_test_missing_offset.db',
            raw_data_file_name=None,
            delete_archive=True)

        with daq:
            time.sleep(0.1)

        self.assertFalse(daq.is_calibrated)
        self.assertEqual(daq.offset, None)
        daq.cleanup()

    def test_zero_offset(self):
        """Test offset value override"""

        channels = ['ch1', 'ch2', 'TRG']
        sample_hz = 100
        trigger_at = 10
        num_records = 500
        n_channels = len(channels) - 1

        data = [mock_record(n_channels) + [0 if (i + 1) < trigger_at else 1]
                for i in range(num_records)]

        device = _MockDevice(data=data, channels=channels, fs=sample_hz)
        daq = DataAcquisitionClient(device=device,
                                    buffer_name='buffer_client_test_offset.db',
                                    raw_data_file_name=None,
                                    delete_archive=True,
                                    clock=CountClock())

        daq.is_calibrated = True  # force the override.
        daq.start_acquisition()
        time.sleep(0.1)
        daq.stop_acquisition()

        self.assertTrue(daq.is_calibrated)
        self.assertEqual(daq.offset, 0.0, "Setting the is_calibrated to True\
            should override the offset calcution.")

        daq.cleanup()

    def test_get_data_for_clock(self):
        """Test queries to the data store by experiment clock units."""

        channels = ['ch1', 'ch2', 'TRG']
        sample_hz = 100
        trigger_at = 10
        num_records = 1000
        n_channels = len(channels) - 1
        data = [mock_record(n_channels) + [0 if (i + 1) < trigger_at else 1]
                for i in range(num_records)]

        device = _MockDevice(data=data, channels=channels, fs=sample_hz)
        daq = DataAcquisitionClient(
            device=device,
            buffer_name='buffer_client_test_get_data_for_clock.db',
            raw_data_file_name=None,
            delete_archive=True,
            clock=CountClock())

        daq.start_acquisition()
        time.sleep(0.2)
        daq.stop_acquisition()

        self.assertTrue(daq.is_calibrated)

        # rownum at calibration should be trigger_at
        self.assertEqual(trigger_at, daq.record_at_calib.rownum)
        self.assertEqual(trigger_at, daq.record_at_calib.timestamp)
        self.assertEqual(data[trigger_at - 1], daq.record_at_calib.data)

        # Test with clocks exactly synchronized.
        self.assertEqual(0.1, daq.offset)
        data_slice = daq.get_data_for_clock(calib_time=0.1, start_time=0.2,
                                            end_time=0.3)
        self.assertEqual(10, len(data_slice))

        start_offset = 20
        for i, record in enumerate(data_slice):
            # mock-data is 0-based, so we have to subtract 1 from the start.
            mock_data_index = i + start_offset - 1
            self.assertEqual(record.data, data[mock_data_index])

        # Test with clocks offset
        data_slice = daq.get_data_for_clock(calib_time=0.2, start_time=0.4,
                                            end_time=0.6)
        self.assertEqual(20, len(data_slice))
        start_offset = 30
        for i, record in enumerate(data_slice):
            # mock-data is 0-based, so we have to subtract 1 from the start.
            mock_data_index = i + start_offset - 1
            self.assertEqual(record.data, data[mock_data_index])

        daq.cleanup()


if __name__ == '__main__':
    unittest.main()
