import unittest
import numpy as np
from bcipy.helpers.load import load_json_parameters
from bcipy.signal.evaluate.rules import Rule, HighVoltage, LowVoltage


class TestRules(unittest.TestCase):

    """Test Rules init and class methods """

    def setUp(self):
        """Create rule objects to test """

        params_file = 'bcipy/parameters/parameters.json'

        self.parameters = load_json_parameters(params_file)

        """Set thresholds for testing """
        self.parameters['highvoltage_value'] = 1
        self.parameters['lowhvoltage_value'] = -1

        self.highvoltage_rule = HighVoltage('highvoltage_threshold',
                                            self.parameters['highvoltage_value'])
        self.lowvoltage_rule = LowVoltage('lowvoltage_threshold',
                                          self.parameters['lowhvoltage_value'])

    def test_rule_init(self):
        """Test that names and thresholds init correctly """

        self.assertEqual(self.highvoltage_rule.name, 'highvoltage_threshold')
        self.assertEqual(self.highvoltage_rule.threshold,
                         self.parameters['highvoltage_value'])

        self.assertEqual(self.lowvoltage_rule.name, 'lowvoltage_threshold')
        self.assertEqual(self.lowvoltage_rule.threshold,
                         self.parameters['lowhvoltage_value'])

        self.assertIsInstance(self.lowvoltage_rule, Rule)
        self.assertIsInstance(self.highvoltage_rule, Rule)

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
