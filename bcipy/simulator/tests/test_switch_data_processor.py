"""Tests for Button Press data processor."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bcipy.acquisition.datastream.mock.switch import switch_device
from bcipy.core.raw_data import RawData
from bcipy.display.main import ButtonPressMode
from bcipy.io.load import load_json_parameters
from bcipy.signal.model.base_model import SignalModelMetadata
from bcipy.signal.model.switch_model import SwitchModel
from bcipy.simulator.data.switch_data_processor import SwitchDataProcessor


class SwitchProcessorTest(unittest.TestCase):
    """Tests for Switch data processor."""

    def setUp(self):
        """Override."""
        self.data_dir = f"{os.path.dirname(__file__)}/resources/"
        self.temp_dir = tempfile.mkdtemp()
        self.model = SwitchModel()
        self.model.metadata = SignalModelMetadata(device_spec=switch_device(),
                                                  evidence_type="BTN",
                                                  transform=None)
        self.params_path = Path(self.data_dir, "btn_subset_parameters.json")

    def test_extracted_data(self):
        """Test extracted data."""
        processor = SwitchDataProcessor(model=self.model)
        params = load_json_parameters(self.params_path)

        extracted_data = processor.process(self.data_dir, params)
        self.assertEqual(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
            extracted_data.labels)
        self.assertEqual((1, 2, 10), extracted_data.inquiries.shape)
        self.assertEqual((1, 20, 1), extracted_data.trials.shape)

        data = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        expected_inquiries = data.reshape((1, 2, 10))
        expected_trials = data.reshape((1, 20, 1))
        self.assertTrue(
            np.allclose(expected_inquiries, extracted_data.inquiries))
        self.assertTrue(np.allclose(expected_trials, extracted_data.trials))

    def test_data_values(self):
        """Test data values with different button press modes"""
        processor = SwitchDataProcessor(model=self.model)
        sample_data = RawData(daq_type='Switch',
                              sample_rate=0,
                              columns=['timestamp', 'Marker', 'lsl_timestamp'])
        sample_data.append([1, 1.0, 6409.33])

        self.assertEqual(
            1.0,
            processor.data_value(sample_data,
                                 inq_start=6408.20,
                                 inq_stop=6412.91,
                                 button_press_mode=ButtonPressMode.ACCEPT))

        self.assertEqual(
            0.0,
            processor.data_value(sample_data,
                                 inq_start=6400.20,
                                 inq_stop=6406.91,
                                 button_press_mode=ButtonPressMode.ACCEPT))

        self.assertEqual(
            0.0,
            processor.data_value(sample_data,
                                 inq_start=6408.20,
                                 inq_stop=6412.91,
                                 button_press_mode=ButtonPressMode.REJECT))

        self.assertEqual(
            1.0,
            processor.data_value(sample_data,
                                 inq_start=6400.20,
                                 inq_stop=6406.91,
                                 button_press_mode=ButtonPressMode.REJECT))


if __name__ == '__main__':
    unittest.main()
