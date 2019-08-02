import unittest
import numpy as np
import math
from bcipy.signal.evaluate.evaluator import Evaluator
from bcipy.helpers.load import load_json_parameters
from bcipy.signal.evaluate.rules import HighVoltage, LowVoltage


class TestEvaluator(unittest.TestCase):
    """Test Evaluator init and class methods """

    def setUp(self):
        self.expected_high_voltage = True
        self.expected_low_voltage = True
        self.parameters = {
            'high_voltage_value': 1,
            'low_voltage_value': -1
        }

        self.evaluator = Evaluator(self.parameters,
                                   self.expected_high_voltage,
                                   self.expected_low_voltage)

    def test_init_rules(self):
        """Test init of ruleset. We expect that rules enabled in the
        params are initialized as part of the ruleset """

        if self.expected_high_voltage:
            self.assertIsInstance(self.evaluator.rules[0], HighVoltage)

        if self.expected_low_voltage:
            self.assertIsInstance(self.evaluator.rules[1], LowVoltage)

    def test_evaluate_with_single_data_point(self):
        """Test evaluate signal with a single data point"""

        # First test rule breakage 
        high_sample = [math.inf]
        low_sample = [-math.inf]
        self.assertFalse(self.evaluator.evaluate(high_sample))
        self.assertFalse(self.evaluator.evaluate(low_sample))

        # Then test rule pass 
        passing_sample = [0]
        self.assertIsNone(self.evaluator.evaluate(passing_sample))

    def test_evaluate_with_array(self):
        """Test evaluate signal with a numpy array """

        # First test rule breakage 
        high_sample_array = np.ones(5) * 5
        low_sample_array = np.ones(5) * -5
        self.assertFalse(self.evaluator.evaluate(high_sample_array))
        self.assertFalse(self.evaluator.evaluate(low_sample_array))

        # Then test rule pass 
        passing_sample_array = np.ones(5) * 0
        self.assertIsNone(self.evaluator.evaluate(passing_sample_array))


if __name__ == "__main__":
    unittest.main()
