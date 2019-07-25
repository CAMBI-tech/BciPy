"""rules.py"""

import numpy as np
from abc import ABC, abstractmethod


class Rule(ABC):

	"""Python Abstract Base Class for a Rule.
	(https://docs.python.org/3/library/abc.html)
	Each rule has a 'isBroken' method which acts as instructions for 
	the evaluator to use when evaluating data. Returns True upon 
	rule breakage, otherwise defaults to False."""

	def __init__(self,name,threshold):

		self.name = name
		
		self.threshold = threshold

	#a method required for all subclasses
	@abstractmethod 
	def isBroken(self,data):

		return False


class HighVoltage(Rule):

	"""High Voltage Rule. Set high threshold for permitted voltage. 
	Separated from LowVoltage for rule specification. Allows
	different types of rules to be fed to artifact rejector
	so that experimenter can differentiate between
	high and low warnings easily. Makes it so that rules 
	are singularly defined. Names must be equal to value in parameters.json"""

	def isBroken(self,data):

		"""
			Test data against threshold value. Return broken
			if threshold exceeded.

			data(ndarray[float]): 1 x N length array where N is the number of samples

              	np.amax takes the maximum value in an array even of length 1:
              	(https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html)
			
		 """

		if  np.amax(data) >= self.threshold:

			return True

		return False


class LowVoltage(Rule):

	"""Low Voltage Rule. Set low threshold for permitted voltage. """

	def isBroken(self,data):

		"""
			Test data against threshold value. Return false
			if threshold exceeded.

			data(ndarray[float]): C x L eeg data where C is number of
                channels and L is the signal length, after filtering


                np.amin takes the minimum value in an array even of length 1:
              	(https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html)
		 """

		if np.amin(data) <= self.threshold:

			return True

		return False

