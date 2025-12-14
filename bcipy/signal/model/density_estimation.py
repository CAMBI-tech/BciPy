import logging
from typing import Optional

import numpy as np
from scipy.stats import iqr
from sklearn.neighbors import KernelDensity

from bcipy.config import SESSION_LOG_FILENAME


class KernelDensityEstimate:
    """Kernel density estimate using scikit learn.
    Attr:
        bandwidth(float): bandwidth of the kernel
        scores(np.array): Shape (num_items, 2) - ratio of classification scores from RDA; used to compute bandwidth
        kernel(string): element to form the actual fitted pdf.
    """

    def __init__(self, scores: Optional[np.array] = None, kernel="gaussian", num_cls=2):
        bandwidth = 1.0 if scores is None else self._compute_bandwidth(
            scores, scores.shape[0])
        self.logger = logging.getLogger(SESSION_LOG_FILENAME)
        self.logger.info(f"KDE. bandwidth={bandwidth}, kernel={kernel}")
        self.num_cls = num_cls
        self.list_den_est = [KernelDensity(
            bandwidth=bandwidth, kernel=kernel) for _ in range(self.num_cls)]

    def _compute_bandwidth(self, scores: np.array, num_items: int):
        """Estimate bandwidth parameter using Silverman's rule of thumb.
        See https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator

        Args:
            scores (np.array): Shape (num_items, 2) - positive and negative class probabilities from RDA
            num_channels (int): number of channels in the original data

        Returns:
            float: rule-of-thumb bandwidth parameter for KDE
        """
        bandwidth = 0.9 * min(np.std(scores), iqr(scores) /
                              1.34) * np.power(num_items, -0.2)
        return bandwidth

    def fit(self, x, y):
        """Fits the kernel density estimates base on labelled data.

        Args:
            x(ndarray[float]): shape (N) data array
            y(ndarray[float]): shape (N) label array
            Where N and c denotes number of samples and classes
            respectively.
        """
        for i, c in enumerate(np.unique(y)):
            dat = x[y == c]
            # Reshape is required, otherwise there's ambiguity if it's one
            # sample with N dims or N samples with 1 dims
            dat = np.squeeze(dat)
            dat = np.expand_dims(dat, axis=1)
            self.list_den_est[i].fit(dat)

    def transform(self, x):
        """Calculates likelihood ods of given data.
        Args:
            x(ndarray[float]): N x 1 data array
            Where N and c denotes number of samples and classes
        Returns:
             val(ndarray[float]): N x c  log-likelihood array
         respectively.
        """

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
