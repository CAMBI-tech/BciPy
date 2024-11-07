import unittest
import shutil
import zipfile
from pathlib import Path
import tempfile

from matplotlib import pyplot as plt

from mockito import when, unstub, verify

from bcipy.helpers.offset import (
    calculate_latency,
    sample_to_seconds,
    extract_data_latency_calculation,
    sample_rate_diffs,
    lsl_timestamp_diffs
)
from bcipy.io.load import load_raw_data
from bcipy.data.raw_data import RawData
from bcipy.data.triggers import trigger_decoder, TriggerType
from bcipy.config import RAW_DATA_FILENAME, TRIGGER_FILENAME

pwd = Path(__file__).absolute().parent
input_folder = pwd / "resources/mock_offset/time_test_data/"


class TestOffset(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.raw_data_path = self.tmp_dir / f'{RAW_DATA_FILENAME}.csv'
        self.triggers_path = str(self.tmp_dir / f'{TRIGGER_FILENAME}')
        with zipfile.ZipFile(input_folder / 'raw_data.csv.zip', 'r') as zip_ref:
            zip_ref.extractall(self.tmp_dir)
        with zipfile.ZipFile(input_folder / 'triggers.txt.zip', 'r') as zip_ref:
            zip_ref.extractall(self.tmp_dir)
        self.raw_data = load_raw_data(self.raw_data_path)
        trigger_targetness, trigger_time, trigger_label = trigger_decoder(
            self.triggers_path,
            remove_pre_fixation=False,
            exclusion=[TriggerType.FIXATION],
            device_type='EEG')
        self.triggers = list(zip(trigger_label, trigger_targetness, trigger_time))
        self.diode_channel = 'TRG'
        self.stim_number = 10

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)
        unstub()
        return super().tearDown()

    def test_sample_to_seconds(self):
        self.assertEqual(sample_to_seconds(100, 1), 0.01)
        self.assertEqual(sample_to_seconds(100, 100), 1)
        self.assertEqual(sample_to_seconds(200, 100), 0.5)

    def test_sample_to_seconds_throws_error_on_zero_args(self):
        with self.assertRaises(AssertionError):
            sample_to_seconds(100, 0)

        with self.assertRaises(AssertionError):
            sample_to_seconds(0, 100)

    def test_extract_data_latency_calculation(self):
        static_offset = 0.0
        recommend = False
        resp = extract_data_latency_calculation(self.tmp_dir, recommend, static_offset)
        self.assertIsInstance(resp[0], RawData)
        self.assertEqual(resp[1], self.triggers)
        self.assertEqual(resp[2], static_offset)

    def test_extract_data_latency_calculation_resets_static_offset_on_recommend(self):
        static_offset = 1.0
        recommend = True
        resp = extract_data_latency_calculation(self.tmp_dir, recommend, static_offset)
        self.assertIsInstance(resp[0], RawData)
        self.assertEqual(resp[1], self.triggers)
        self.assertNotEqual(resp[2], static_offset)
        self.assertEqual(resp[2], 0.0)

    def test_sample_rate_diffs(self):
        response = sample_rate_diffs(self.raw_data)
        self.assertIsInstance(response, tuple)
        self.assertEqual(int(response[0]), int(response[1]))

    def test_lsl_timestamp_diffs(self):
        response = lsl_timestamp_diffs(self.raw_data)
        self.assertIsInstance(response, list)

    def test_calculate_latency_defaults(self):
        response = calculate_latency(
            self.raw_data,
            self.diode_channel,
            self.triggers,
            self.stim_number)
        self.assertIsInstance(response, tuple)
        self.assertIsInstance(response[0], list)
        self.assertIsInstance(response[1], list)
        # check that error response is not empty
        self.assertTrue(len(response[1]) > 1)

    def test_calculate_latency_with_high_tolerance(self):
        response = calculate_latency(
            self.raw_data,
            self.diode_channel,
            self.triggers,
            self.stim_number,
            tolerance=0.5)
        self.assertIsInstance(response, tuple)
        self.assertIsInstance(response[0], list)
        self.assertIsInstance(response[1], list)
        # check that given the high tolerance, the error response is empty for this data
        self.assertEqual(len(response[1]), 0.0)

    def test_calculate_latency_with_recommend(self):
        recommend = True
        response = calculate_latency(
            self.raw_data,
            self.diode_channel,
            self.triggers,
            self.stim_number,
            recommend_static=recommend)
        self.assertIsInstance(response, tuple)
        self.assertIsInstance(response[0], list)
        self.assertIsInstance(response[1], list)
        self.assertEqual(len(response[1]), 0.0)

    def test_calculate_latency_ploting(self):
        # stub the plot show function before calling
        when(plt).show().thenReturn(None)

        calculate_latency(
            self.raw_data,
            self.diode_channel,
            self.triggers,
            self.stim_number,
            plot=True)

        # verify that the plot show function was called 3 times as expected
        verify(plt, times=3).show()

    def test_calculate_latency_ploting_with_false_positive_correction(self):
        """Test calculate_latency with false positive correction."""
        corrections = 2

        response1 = calculate_latency(
            self.raw_data,
            self.diode_channel,
            self.triggers,
            self.stim_number)

        response2 = calculate_latency(
            self.raw_data,
            self.diode_channel,
            self.triggers,
            self.stim_number,
            correct_diode_false_positive=corrections)

        # verify two responses are different in the expected way
        self.assertNotEqual(response1[0], response2[0])
        self.assertEqual(len(response2[0]), len(response1[0]) - corrections)
        self.assertNotEqual(response1[1][0], response2[1][0])


if __name__ == "__main__":
    unittest.main()
