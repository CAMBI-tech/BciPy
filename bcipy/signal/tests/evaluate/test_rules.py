import unittest
import numpy as np
from bcipy.helpers.load import load_json_parameters
from bcipy.signal.evaluate.rules import Rule, HighVoltage, LowVoltage


class TestRules(unittest.TestCase):
    """Test Rules init and class methods """

    def setUp(self):
        """Create rule objects to test """

        # Set thresholds for testing
        self.highvoltage_value = 1
        self.lowvoltage_value = -1

        self.highvoltage_rule = HighVoltage(self.highvoltage_value)
        self.lowvoltage_rule = LowVoltage(self.lowvoltage_value)

    def test_high_voltage_rule_init(self):
        """Test that high voltage inits correctly"""
        self.assertEqual(self.highvoltage_rule.threshold,
                         self.highvoltage_value)

        self.assertIsInstance(self.highvoltage_rule, Rule)

    def test_low_voltage_rule_init(self):
        """Test that low voltage inits correctly"""
        self.assertEqual(self.lowvoltage_rule.threshold,
                         self.lowvoltage_value)

        self.assertIsInstance(self.lowvoltage_rule, Rule)

    def test_lowvoltage_on_single_datapoint(self):
        """Test passing and failing samples for lowvoltage """

        failing_sample = -2
        passing_sample = 0.2
        self.assertTrue(self.lowvoltage_rule.is_broken(failing_sample))
        self.assertFalse(self.lowvoltage_rule.is_broken(passing_sample))

    def test_lowvoltage_on_array(self):
        """Test passing and failing arrays for lowvoltage """

        failing_array = np.ones(5) * -1.5
        passing_array = np.ones(5) / 2
        self.assertTrue(self.lowvoltage_rule.is_broken(failing_array))
        self.assertFalse(self.lowvoltage_rule.is_broken(passing_array))

    def test_highvoltage_on_single_datapoint(self):
        """Test passing and failing samples for highvoltage """

        failing_sample = 2
        passing_sample = 0.2
        self.assertTrue(self.highvoltage_rule.is_broken(failing_sample))
        self.assertFalse(self.highvoltage_rule.is_broken(passing_sample))

    def test_highvoltage_on_array(self):
        """Test passing and failing arrays for highvoltage """

        failing_array = np.ones(5) * 1.5
        passing_array = np.ones(5) / 2
        self.assertTrue(self.highvoltage_rule.is_broken(failing_array))
        self.assertFalse(self.highvoltage_rule.is_broken(passing_array))


if __name__ == "__main__":
    unittest.main()
