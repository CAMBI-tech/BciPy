# pylint: disable=no-self-use
"""Tests for the processor module."""
import unittest
import pytest
from mock import mock_open, patch
from mockito import any, mock, unstub, verify, when
from bcipy.acquisition.device_info import DeviceInfo
from bcipy.acquisition.processor import FileWriter, DispatchProcessor, Processor


class TestFilewriter(unittest.TestCase):
    """Tests for the Processor that writes the rawdata files."""

    def test_filewriter(self):
        """Test FileWriter functionality"""

        data = [[i + j for j in range(3)] for i in range(3)]
        expected_csv_rows = ['0,1,2\r\n', '1,2,3\r\n', '2,3,4\r\n']

        filewriter = FileWriter('foo.csv')
        filewriter.set_device_info(DeviceInfo(name='foo-device', fs=100,
                                              channels=['c1', 'c2', 'c3']))

        mockopen = mock_open()
        with patch('bcipy.acquisition.processor.open', mockopen):
            with filewriter:
                mockopen.assert_called_once_with('foo.csv', 'w', newline='')

                handle = mockopen()
                handle.write.assert_called_with('timestamp,c1,c2,c3\r\n')

                for i, row in enumerate(data):
                    timestamp = float(i)
                    filewriter.process(row, timestamp)
                    handle.write.assert_called_with(
                        str(timestamp) + "," + str(expected_csv_rows[i]))

            mockopen().close.assert_called_once()

    def test_filewriter_setup(self):
        """
        Test that FileWriter throws an exception if it is used without setting
        the device_info.
        """

        filewriter = FileWriter('foo.csv')

        with pytest.raises(Exception):
            with filewriter:
                pass


class TestDispatchProcessor(unittest.TestCase):
    """Tests for the MultiProcessor."""

    def test_set_device_info(self):
        """Test MultiProcessor functionality"""

        proc1 = mock(Processor)
        proc2 = mock(Processor)
        proc3 = mock(Processor)

        when(proc1).set_device_info(any()).thenReturn(None)
        when(proc2).set_device_info(any()).thenReturn(None)
        when(proc3).set_device_info(any()).thenReturn(None)

        multi = DispatchProcessor(proc1, proc2)

        device_info = DeviceInfo(name='foo-device', fs=100,
                                 channels=['c1', 'c2', 'c3'])

        multi.set_device_info(device_info)
        verify(proc1, times=1).set_device_info(device_info)
        verify(proc2, times=1).set_device_info(device_info)

        multi.add(proc3)
        verify(proc3, times=1).set_device_info(device_info)

    def test_process(self):
        """Test MultiProcessor functionality"""

        data1 = [1,2,3]
        data2 = [4,5,6]

        proc1 = mock(Processor)
        proc2 = mock(Processor)

        when(proc1).process(any(), None).thenReturn(None)
        when(proc2).process(any(), None).thenReturn(None)

        multi = DispatchProcessor(proc1)
        multi.process(data1)

        multi.add(proc2)
        multi.process(data2)

        verify(proc1, times=1).process(data1, None)
        verify(proc1, times=1).process(data2, None)
        verify(proc2, times=0).process(data1, None)
        verify(proc2, times=1).process(data2, None)

