"""Tests for Button Press data processor."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bcipy.acquisition.datastream.mock.switch import switch_device
from bcipy.io.load import load_json_parameters
from bcipy.signal.model.base_model import SignalModelMetadata
from bcipy.simulator.demo.button_press_data_processor import \
    ButtonPressDataProcessor
from bcipy.simulator.demo.button_press_model import ButtonPressModel


class ButtonPressProcessorTest(unittest.TestCase):
    """Tests for Button Press data processor."""

    def setUp(self):
        """Override."""
        self.data_dir = f"{os.path.dirname(__file__)}/resources/"
        self.temp_dir = tempfile.mkdtemp()
        self.model = ButtonPressModel()
        self.model.metadata = SignalModelMetadata(device_spec=switch_device(),
                                                  evidence_type="BTN",
                                                  transform=None)

        self.params_path = Path(self.data_dir, "btn_subset_parameters.json")

    def test_extracted_data(self):
        """Test extracted data."""
        processor = ButtonPressDataProcessor(model=self.model)
        params = load_json_parameters(self.params_path)

        extracted_data = processor.process(self.data_dir, params)
        self.assertEqual(
            [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            extracted_data.labels)
        self.assertEqual((1, 2, 10), extracted_data.inquiries.shape)
        self.assertEqual((1, 20, 1), extracted_data.trials.shape)

        data = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        expected_inquiries = data.reshape((1, 2, 10))
        expected_trials = data.reshape((1, 20, 1))
        self.assertTrue(
            np.allclose(expected_inquiries, extracted_data.inquiries))
        self.assertTrue(np.allclose(expected_trials, extracted_data.trials))


if __name__ == '__main__':
    unittest.main()
