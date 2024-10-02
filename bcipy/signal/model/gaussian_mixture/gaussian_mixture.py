from pathlib import Path

import numpy as np
from bcipy.signal.model import SignalModel
from sklearn.mixture import GaussianMixture
from bcipy.helpers.stimuli import GazeReshaper
from sklearn.model_selection import cross_val_score  # noqa
import scipy.stats as stats

from typing import Optional


class GazeModelIndividual(SignalModel):
    reshaper = GazeReshaper()

    def __init__(self, num_components=2):
        self.num_components = num_components   # number of gaussians to fit

    def fit(self, train_data: np.ndarray) -> 'GazeModelIndividual':
        model = GaussianMixture(n_components=1, random_state=0, init_params='kmeans')
        model.fit(train_data)
        self.model = model

        return self

    def evaluate(self, test_data: np.ndarray):
        '''
        Compute mean and covariance of each mixture component.
        '''

        means = self.model.means_
        covs = self.model.covariances_

        return means, covs

    def predict(self, test_data: np.ndarray, means: np.ndarray, covs: np.ndarray):
        '''
        Compute log-likelihood of each sample.
        Predict the labels for the test data.
        '''

        num_components = len(means)

        N, D = test_data.shape
        K = num_components

        likelihoods = np.zeros((N, K), dtype=object)
        predictions = np.zeros(N, dtype=object)

        # Find the likelihoods by insterting the test data into the pdf of each component
        for i in range(N):
            for k in range(K):
                mu = means[k]
                sigma = covs[k]

                likelihoods[i, k] = stats.multivariate_normal.pdf(test_data[i], mu, sigma)

            # Find the argmax of the likelihoods to get the predictions
            predictions[i] = np.argmax(likelihoods[i])

        return likelihoods, predictions

    def calculate_acc(self, test_data: np.ndarray, test_labels: Optional[np.ndarray]):
        '''
        Compute model performance characteristics on the provided test data and labels.
        '''

        # return accuracy

    def save(self, path: Path):
        """Save model state to the provided checkpoint"""
        ...

    def load(self, path: Path):
        """Load model state from the provided checkpoint"""
        ...


class GazeModelCombined(SignalModel):
    '''Gaze model that uses all symbols to fit a single Gaussian '''
    reshaper = GazeReshaper()

    def __init__(self, num_components=1):
        self.num_components = num_components   # number of gaussians to fit

    def fit(self, train_data: np.ndarray):
        model = GaussianMixture(n_components=1, random_state=0, init_params='kmeans')
        model.fit(train_data)
        self.model = model

        return self

    def evaluate(self, test_data: np.ndarray, sym_pos: np.ndarray):
        '''
        Return mean and covariance of each mixture component.

        test_data: gaze data for each symbol
        sym_pos: mid positions for each symbol in Tobii coordinates
        '''

        means = self.model.means_ + sym_pos
        covs = self.model.covariances_

        return means, covs

    def predict(self, test_data: np.ndarray):
        '''
        Compute log-likelihood of each sample.
        Predict the labels for the test data.
        '''
        # Compute over log-likelihood scores
        # Get the argmax of the scores

        scores = self.model.score_samples(test_data)

        # return predictions

    def calculate_acc(self, test_data: np.ndarray, sym_pos: np.ndarray):
        '''
        Compute model performance characteristics on the provided test data and labels.
        '''

        # return accuracy

    def save(self, path: Path):
        """Save model state to the provided checkpoint"""
        ...

    def load(self, path: Path):
        """Load model state from the provided checkpoint"""
        ...
