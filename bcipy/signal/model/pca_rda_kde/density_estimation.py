from typing import Optional

import numpy as np

from scipy.stats import iqr
from sklearn.neighbors import KernelDensity
import logging


class KernelDensityEstimate:
    def __init__(self, scores: Optional[np.array] = None,
                 algorithm='auto', kernel='gaussian', metric='euclidean', atol=0, rtol=0,
                 breadth_first=True, leaf_size=40, metric_params=None, num_cls=2):
        """ Kernel density estimate implementation using scikit learn
        library. For further reference, please check scikit learn website.
        Attr:
            bandwidth(float): bandwidth of the kernel
            scores(np.array): Shape (num_items, 2) - ratio of classification scores from RDA; used to compute bandwidth
            algorithm(string): algorithm type
            kernel(string): element to form the actual fitted pdf.
            metric(string): distance metric used by algorithm to insert kernels
                onto samples in the given domain.
            atol(float): absolute tolerance to fit a probability distribution
            rtol(float): relative tolerance
            breadth_first(bool): Flag to use breadth first search
            leaf_size(int): Uses a tree model to fit probability density.
            metric_params(dictionary): Used by tree model
        """
        if scores is None:
            bandwidth = 1.0
        else:
            num_items = scores.shape[0]
            bandwidth = self._compute_bandwidth(scores, num_items)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"bandwidth: {bandwidth}")
        self.list_den_est = []
        self.num_cls = num_cls
        for _ in range(num_cls):
            self.list_den_est.append(KernelDensity(bandwidth=bandwidth,
                                                   algorithm=algorithm,
                                                   kernel=kernel,
                                                   metric=metric,
                                                   atol=atol,
                                                   rtol=rtol,
                                                   breadth_first=breadth_first,
                                                   leaf_size=leaf_size,
                                                   metric_params=metric_params))

    # def _compute_bandwidth(self, scores: np.array, num_channels: int):
    def _compute_bandwidth(self, scores: np.array, num_items: int):
        """Estimate bandwidth parameter

        Args:
            scores (np.array): Shape (num_items, 2) - positive and negative class probabilities from RDA
            num_channels (int): number of channels in the original data

        Returns:
            float: rule-of-thumb bandwidth parameter for KDE
        """
        # bandwidth = 1.06 * min(np.std(scores), iqr(scores) / 1.34) * np.power(num_channels, -0.2)
        bandwidth = 1.06 * min(np.std(scores), iqr(scores) / 1.34) * np.power(num_items, -0.2)
        return bandwidth

    def fit(self, x, y):
        """ Fits the kernel density estimates base on labelled data.
            Attr:
                x(ndarray[float]): N x 1 data array
                y(ndarray[float]): N x 1 label array
                Where N and c denotes number of samples and classes
                respectively. """

        classes = np.unique(y)

        cls_dep_x = [x[np.where(y == classes[i])[0]] for i in
                     range(self.num_cls)]

        for i in range(self.num_cls):
            # Reshape is required, otherwise there's ambiguity if it's one
            # sample with N dims or N samples with 1 dims

            dat = np.squeeze(cls_dep_x[i])
            dat = np.expand_dims(dat, axis=1)

            self.list_den_est[i].fit(dat)

    def transform(self, x):
        """ Calculates likelihood ods of given data.
            Attr:
                x(ndarray[float]): N x 1 data array
                Where N and c denotes number of samples and classes
            Return:
                 val(ndarray[float]): N x c  log-likelihood array
             respectively. """

        # Calculate likelihoods for each density estimate
        val = []
        for i in range(self.num_cls):
            dat = np.squeeze(x)
            dat = np.expand_dims(dat, axis=1)
            val.append(self.list_den_est[i].score_samples(dat))

        return np.transpose(np.array(val))

    def fit_transform(self, x, y):

        self.fit(x, y)
        return self.transform(x)
