import unittest
import numpy as np
from bcipy.signal.evaluate.rules import Rule, HighVoltage, LowVoltage
from bcipy.signal.generator.generator import gen_random_data


class TestRules(unittest.TestCase):
    """Test Rules init and class methods """

    def setUp(self):
        """Create rule objects to test """

        # Set thresholds for testing
        self.highvoltage_value = 1
        self.lowvoltage_value = -1

        self.highvoltage_rule = HighVoltage(self.highvoltage_value)
        self.lowvoltage_rule = LowVoltage(self.lowvoltage_value)

        self.channels = 32

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

    def test_low_voltage_failing_signal(self):
        """Test generated sub threshold signal against low voltage"""
        data = gen_random_data(1,5,self.channels)
        #ascertain that at least one random datapoint is below threshold to test np.amin edgecase
        data[np.random.randint(self.channels)] = -1.5
        self.assertTrue(self.lowvoltage_rule.is_broken(data))

    def test_low_voltage_passing_signal(self):
        """Test generated signal that is consistently above threshold"""
        data = gen_random_data(-0.5,0.5,self.channels)
        self.assertFalse(self.lowvoltage_rule.is_broken(data))

    def test_high_voltage_failing_signal(self):
        """Test generated signal with one data point above threshold """
        data = gen_random_data(-5,0,self.channels)
        #ascertain that at least one random datapoint is above threshold to test np.amax edgecase
        data[np.random.randint(self.channels)] = 1.5
        self.assertTrue(self.highvoltage_rule.is_broken(data))

    def test_high_voltage_passing_signal(self):
        """Test generated signal that is consistently below threshold"""
        data = gen_random_data(-0.5,0.5,self.channels)
        self.assertFalse(self.highvoltage_rule.is_broken(data))

if __name__ == "__main__":
    unittest.main()
