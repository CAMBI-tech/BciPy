""" Wraps the Machine Learning Box which includes a classifier and a
dimensionality reduction method. It should form a pipeline to apply
different kinds of structures one after another.
"""

import sys
import numpy as np

sys.path.append(
    'C:\Users\Aziz\Desktop\GIT\TMbci\eeg_model\mach_learning_dir\classifier')
sys.path.append(
    'C:\Users\Aziz\Desktop\GIT\TMbci\eeg_model\mach_learning_dir\dimensionality_reduction')

from function_classifier import RegularizedDiscriminantAnalysis
from function_dim_reduction import PrincipalComponentAnalysis


class PipeLine(object):
    """ Forms a pipeline using multiple dimensionality reductions and a
    final classifier. Observe that each function should include;
        - fit
        - transform
        - update
        - fit_transform
    Attr:
        pipeline(list[methods]): Wrapper for pipeline structure
        """

    def __init__(self):
        self.pipeline = []

    def add(self, method):
        """ Adds an object to the pipeline.
        Args:
            method(function): dim reduction or classification function """
        self.pipeline.append(method)

    def fit(self, x, y):
        """ Given the observation and the label, trains all elements in the
        pipeline. Observe that each element in the pipeline should have a
        'fit' function.
        Args:
            x(ndarray[float]): of desired shape
            y(ndarray[int]): of desired shape """
        line_el = [x]
        for i in range(len(self.pipeline) - 1):
            line_el.append(self.pipeline[i].fit_transform(line_el[i]))

        self.pipeline[-1].fit(line_el[-1], y)

    def fit_transform(self, x, y):
        """ Applies fit transform on all functions
        Args:
            x(ndarray[float]): of desired shape
            y(ndarray[int]): of desired shape """

        line_el = [x]
        for i in range(len(self.pipeline) - 1):
            line_el.append(self.pipeline[i].fit_transform(line_el[i], y))

        arg = self.pipeline[-1].fit_transform(line_el[-1], y)
        return arg

    def transform(self, x):
        """ Applies transform on all functions. Prior to using transform on
        pipeline, it should be trained.
        Args:
             x(ndarray[float]): of desired shape """
        line_el = [x]
        for i in range(len(self.pipeline)):
            line_el.append(self.pipeline[i].transform(line_el[i]))

        return line_el[-1]
