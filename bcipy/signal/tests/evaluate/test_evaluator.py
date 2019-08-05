import unittest
import numpy as np
import math
from bcipy.signal.evaluate.evaluator import Evaluator
from bcipy.signal.evaluate.rules import HighVoltage, LowVoltage
from mockito import any, unstub, when
from bcipy.signal.generator.generator import gen_random_data


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

        self.channels = 32

    def tearDown(self):
        unstub()

    def test_init_rules(self):
        """Test init of ruleset. We expect that rules enabled in the
        params are initialized as part of the ruleset"""

        if self.expected_high_voltage:
            self.assertIsInstance(self.evaluator.rules[0], HighVoltage)

        if self.expected_low_voltage:
            self.assertIsInstance(self.evaluator.rules[1], LowVoltage)

    def test_evaluate_true(self):
        """Test evaluate with is_broken returning True"""
        when(self.evaluator.rules[0]).is_broken(any()).thenReturn(True)
        when(self.evaluator.rules[1]).is_broken(any()).thenReturn(True)
        data = gen_random_data(-1,2,self.channels)
        self.assertFalse(self.evaluator.evaluate(data))

    def test_evaluate_false(self):
        """Test evaluate with is_broken returning False """
        when(self.evaluator.rules[0]).is_broken(any()).thenReturn(False)
        when(self.evaluator.rules[1]).is_broken(any()).thenReturn(False)
        data = gen_random_data(-1,2,self.channels)
        self.assertIsNone(self.evaluator.evaluate(data))


if __name__ == "__main__":
    unittest.main()
