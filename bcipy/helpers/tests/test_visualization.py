import unittest
import tempfile
import shutil
from pathlib import Path

from mockito import when, any, verify, unstub, mock

from bcipy.helpers.visualization import visualize_session_data
from bcipy.helpers import visualization
from bcipy.helpers.raw_data import RawData
from bcipy.helpers.load import load_json_parameters
from bcipy.config import DEFAULT_PARAMETERS_PATH


class TestVisualizeSessionData(unittest.TestCase):
    """Test Session Data Visualization."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.parameters = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True)
        self.raw_data_mock = mock()
        self.raw_data_mock.daq_type = 'DSI-24'
        self.raw_data_mock.sample_rate = 300
        self.channel_map_mock = mock()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        unstub()

    def test_visualize_session_data(self):
        """Test visualize_session_data with default parameters."""
        trigger_timing_mock = [1, 2]
        trigger_label_mock = ['target', 'nontarget']
        show = True
        when(RawData).load(any()).thenReturn(self.raw_data_mock)
        when(visualization).analysis_channels(any(), any()).thenReturn(self.channel_map_mock)
        when(visualization).trigger_decoder(
            offset=any(), trigger_path=any(), exclusion=any()).thenReturn(
            (trigger_label_mock, trigger_timing_mock, []))
        when(visualization).visualize_erp(
            self.raw_data_mock,
            self.channel_map_mock,
            trigger_timing_mock,
            any(),
            any(),
            transform=any(),
            plot_average=any(),
            plot_topomaps=any(),
            save_path=any(),
            show=show).thenReturn(True)

        # This fuction returns a figure handle directly from visualization.visualize_erp,
        # however, we are mocking it to return True here
        resp = visualize_session_data(self.tmp_dir, self.parameters, show=show)
        self.assertTrue(resp)
        verify(RawData, times=1).load(any())
        verify(visualization, times=1).analysis_channels(any(), any())
        verify(visualization, times=1).trigger_decoder(offset=any(), trigger_path=any(), exclusion=any())
        verify(visualization, times=1).visualize_erp(
            self.raw_data_mock,
            self.channel_map_mock,
            trigger_timing_mock,
            any(),
            any(),
            transform=any(),
            plot_average=any(),
            plot_topomaps=any(),
            save_path=any(),
            show=show)

    def test_visualize_session_data_with_no_valid_targets(self):
        """Test visualize_session_data throws an error if there are no valid targets."""
        trigger_timing_mock = [1, 2]
        # don't include target in the trigger labels
        trigger_label_mock = ['nontarget', 'nontarget']
        show = False
        when(RawData).load(any()).thenReturn(self.raw_data_mock)
        when(visualization).analysis_channels(any(), any()).thenReturn(self.channel_map_mock)
        when(visualization).trigger_decoder(
            offset=any(), trigger_path=any(), exclusion=any()).thenReturn(
            (trigger_label_mock, trigger_timing_mock, []))

        with self.assertRaises(AssertionError):
            visualize_session_data(self.tmp_dir, self.parameters, show=show)

    def test_visualize_session_data_with_no_valid_nontargets(self):
        """Test visualize_session_data throws an error if there are no valid nontargets."""
        trigger_timing_mock = [1, 2]
        # don't include nontarget in the trigger labels
        trigger_label_mock = ['target', 'target']
        show = False
        when(RawData).load(any()).thenReturn(self.raw_data_mock)
        when(visualization).analysis_channels(any(), any()).thenReturn(self.channel_map_mock)
        when(visualization).trigger_decoder(
            offset=any(), trigger_path=any(), exclusion=any()).thenReturn(
            (trigger_label_mock, trigger_timing_mock, []))

        with self.assertRaises(AssertionError):
            visualize_session_data(self.tmp_dir, self.parameters, show=show)

    def test_visualize_session_data_with_invalid_timing(self):
        """Test visualize_session_data throws an error if there are invalid trigger times."""
        # in this case, leave out the second trigger time
        trigger_timing_mock = [1]
        trigger_label_mock = ['target', 'nontarget']
        show = False
        when(RawData).load(any()).thenReturn(self.raw_data_mock)
        when(visualization).analysis_channels(any(), any()).thenReturn(self.channel_map_mock)
        when(visualization).trigger_decoder(
            offset=any(), trigger_path=any(), exclusion=any()).thenReturn(
            (trigger_label_mock, trigger_timing_mock, []))

        with self.assertRaises(AssertionError):
            visualize_session_data(self.tmp_dir, self.parameters, show=show)

        # in this case, add a third trigger time
        trigger_timing_mock = [1, 2, 3]
        when(visualization).trigger_decoder(
            offset=any(), trigger_path=any(), exclusion=any()).thenReturn(
            (trigger_label_mock, trigger_timing_mock, []))

        with self.assertRaises(AssertionError):
            visualize_session_data(self.tmp_dir, self.parameters, show=show)


if __name__ == "__main__":
    unittest.main()
