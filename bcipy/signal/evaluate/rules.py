"""rules.py"""

from bcipy.helpers.load import load_json_parameters
import numpy as np


class Rule:

	"""Abstract base class for a rule. Each rule has a 'test' method
	which acts as instructions for the evaluator to use when evaluating data
	and an isBroken method, which returns True upon rule breakage."""

	def __init__(self,name):

		self.name = name
		
		params_path = 'bcipy/parameters/parameters.json'

		self.threshold = load_json_parameters(params_path,value_cast=True)[self.name]


class HighVoltage(Rule):

	"""High Voltage Rule. Set high threshold for permitted voltage. 
	Separated from LowVoltage for rule specification. Allows
	different types of rules to be fed to artifact rejector
	so that experimenter can differentiate between
	high and low warnings easily. Makes it so that rules 
	are singularly defined. Names must be equal to title in parameters.json"""

	def __init__(self):

		Rule.__init__(self,"HighVoltage Value")

	def isBroken(self,data):

		"""
			Test data against threshold value. Return broken
			if threshold exceeded.

			data(ndarray[float]): C x L eeg data where C is number of
                channels and L is the signal length, after filtering

		 """

		if np.amax(data) >= self.threshold:

			return True


class LowVoltage(Rule):

	"""Low Voltage Rule. Set low threshold for permitted voltage. """

	def __init__(self):

		Rule.__init__(self,"LowVoltage Value")

	def isBroken(self,data):

		"""
			Test data against threshold value. Return false
			if threshold exceeded.

			data(ndarray[float]): C x L eeg data where C is number of
                channels and L is the signal length, after filtering
		 """

		if np.amin(data) <= self.threshold:

			return True

