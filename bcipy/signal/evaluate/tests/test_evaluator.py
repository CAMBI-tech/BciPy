import unittest
import numpy as np
from bcipy.signal.evaluate import Evaluator
from bcipy.signal.evaluate import HighVoltage,LowVoltage

class TestEvaluator(unittest.TestCase):

	"""Test Evaluator init and class methods """

	def setUp(self):

		params_file = 'bcipy/parameters/parameters.json'

        self.parameters = load_json_parameters(params_file,
                                               value_cast=True)

		self.evaluator = Evaluator(parameters)


		self. expected_high_voltage = self.parameters['highvoltage_threshold']
		self. expected_low_voltage = self.parameters['lowvoltage_threshold']

	def test_init(self):

		"""Test init of ruleset, broken rules """

		if expected_high_voltage:
			assertIn(self.evaluator.rules,HighVoltage())

		if expected_low_voltage:
			assertIn(self.evaluator.rules,LowVoltage())

		for element in self.evaluator.broken_rules.values():
			assertFalse(element)


	def test_evaluate_with_single_data_point(sample):

		"""Test evaluate signal with a single data point"""

		sample = [50]

	def test_evaluate_with_array(array):

		"""Test evaluate signal with a numpy array """

		pass

if __name__ == "__main__":

	pass