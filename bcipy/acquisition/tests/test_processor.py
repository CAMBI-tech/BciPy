# pylint: disable=no-self-use
"""Tests for the processor module."""
import unittest
from mockito import any, mock, verify, when
from bcipy.acquisition.device_info import DeviceInfo
from bcipy.acquisition.processor import DispatchProcessor, Processor


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

        device_info = DeviceInfo(name='foo-device',
                                 fs=100,
                                 channels=['c1', 'c2', 'c3'])

        multi.set_device_info(device_info)
        verify(proc1, times=1).set_device_info(device_info)
        verify(proc2, times=1).set_device_info(device_info)

        multi.add(proc3)
        verify(proc3, times=1).set_device_info(device_info)

    def test_process(self):
        """Test MultiProcessor functionality"""

        data1 = [1, 2, 3]
        data2 = [4, 5, 6]

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
