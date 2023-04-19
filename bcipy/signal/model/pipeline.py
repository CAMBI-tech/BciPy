""" Wraps the Machine Learning Box which includes a classifier and a
dimensionality reduction method. It should form a pipeline to apply
different kinds of structures one after another.
"""


class Pipeline(object):
    """ Forms a pipeline using multiple dimensionality reductions and a
    final classifier. Observe that each function should include;
        - fit
        - transform
        - update
        - fit_transform
    Attr:
        pipeline(list[methods]): wrapper for pipeline structure
        line_el(list[ndarray[float]]): values at each input.
            [0] is the input and the rest follows
        """

    def __init__(self, pipeline=None):
        self.pipeline = pipeline if pipeline is not None else []
        self.line_el = []

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
        self.line_el = [x]
        for i in range(len(self.pipeline) - 1):
            self.line_el.append(self.pipeline[i].fit_transform(self.line_el[i], y))

        self.pipeline[-1].fit(self.line_el[-1], y)

    def fit_transform(self, x, y):
        """ Applies fit transform on all functions
        Args:
            x(ndarray[float]): of desired shape
            y(ndarray[int]): of desired shape """

        self.line_el = [x]
        for i in range(len(self.pipeline) - 1):
            self.line_el.append(self.pipeline[i].fit_transform(self.line_el[i], y))

        arg = self.pipeline[-1].fit_transform(self.line_el[-1], y)
        return arg

    def transform(self, x):
        """ Applies transform on all functions. Prior to using transform on
        pipeline, it should be trained.
        Args:
             x(ndarray[float]): of desired shape """
        self.line_el = [x]
        for i in range(len(self.pipeline)):
            self.line_el.append(self.pipeline[i].transform(self.line_el[i]))

        return self.line_el[-1]
