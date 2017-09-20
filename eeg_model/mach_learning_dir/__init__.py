""" Wraps the Machine Learning Box which includes a classifier and a
dimensionality reduction method. It should form a pipeline to apply
different kinds of structures one after another.
"""

import sys
import numpy as np

sys.path.append('.\classifier')
sys.path.append('.\dimensionality_reduction')

from function_classifier import RegularizedDiscriminantAnalysis
from function_dim_reduction import PrincipalComponentAnalysis


class PipelineWrapper(object):
    """ Forms a pipeline using multiple dimensionality reductions and a
    final classifier. Observe that each function should include;
        - fit
        - transform
        - update
        - fit_transform
        - opt_param
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
            x(ndarray[float]): N x k data array
            y(ndarray[int]): N x k observation (class) array
                N is number of samples k is dimensionality of features """
        line_el = [x]
        for i in range(len(self.pipeline) - 1):
            line_el.append(self.pipeline[i].fit_transform(line_el[i]))

        self.pipeline[-1].fit_param(line_el[-1], y)

    def fit_transform(self, x, y):
        """ Applies fit transform on all functions
        Args:
            x(ndarray[float]): N x k data array
            y(ndarray[int]): N x k observation (class) array
                N is number of samples k is dimensionality of features  """
        line_el = [x]
        for i in range(len(self.pipeline) - 1):
            line_el.append(self.pipeline[i].fit_transform(line_el[i]))

        arg = self.pipeline[-1].fit_transform(line_el[-1], y)
        return arg

    def transform(self, x):
        """ Applies transform on all functions. Prior to using transform on
        pipeline, it should be trained.
        Args:
            x(ndarray[float]): N x k data array
                N is number of samples k is dimensionality of features  """
        line_el = [x]
        for i in range(len(self.pipeline)):
            line_el.append(self.pipeline[i].transform(line_el[i]))

        return line_el[-1]


def main(x, y=None, model=None):
    # TODO: Think about I/O models of the machine learning fun.
    if not model:
        # TODO: Get parameters about the pipeline from the parameter folder
        # PCA and RDA are not from a black box, they should be parametrized.
        model = PipelineWrapper()
        model.add(PrincipalComponentAnalysis)
        model.add(RegularizedDiscriminantAnalysis)
        arg = model.fit_transform(x, y)
    else:
        arg = model.transform(x)

    return [arg, model]


if __name__ == "__main__": main()
