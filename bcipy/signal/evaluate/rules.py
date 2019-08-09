import numpy as np
from abc import ABC, abstractmethod


class Rule(ABC):
    """Rule.

    Python Abstract Base Class for a Rule.
    (https://docs.python.org/3/library/abc.html)
    Each rule has a 'is_broken' method which acts as instructions for
    the evaluator to use when evaluating data. Returns True upon
    rule breakage, otherwise defaults to False."""

    def __init__(self, threshold):
        self.threshold = threshold

    # a method required for all subclasses
    @abstractmethod
    def is_broken(self, data):
        ...


class HighVoltage(Rule):
    """High Voltage Rule.

    Set high threshold for permitted voltage.

    Allows different types of rules to be fed to artifact rejector so that experimenter can differentiate between
    high and low warnings easily. Makes it so that rules are singularly defined. Names must be equal to value in parameters.json
    """

    def is_broken(self, data):
        """Is Broken.

            Test data against threshold value. Return broken
            if threshold exceeded.

            data(ndarray[float]): C x N length array where
            C is the number of channels and N is the number of samples

                np.amax takes the maximum value in an
                array even of length 1:
                (https://docs.scipy.org/doc/numpy/
                reference/generated/numpy.amax.html)
        """
        if np.amax(data) >= self.threshold:
            return True

        return False
    
    def __str__(self):
        return f'High Voltage with threshold {self.threshold}'


class LowVoltage(Rule):
    """Low Voltage Rule.
    
    Set low threshold for permitted voltage.
    """

    def is_broken(self, data):
        """Is Broken.

            Test data against threshold value. Return false
            if threshold exceeded.


            data(ndarray[float]): C x N length array where
            C is the number of channels and N is the number of samples


                np.amin takes the minimum value in an array even of length 1:
                  (https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html)
         """
        if np.amin(data) <= self.threshold:
            return True

        return False

    def __str__(self):
        return f'Low Voltage with threshold {self.threshold}'
