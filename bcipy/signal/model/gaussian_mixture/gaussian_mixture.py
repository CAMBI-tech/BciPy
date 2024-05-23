from pathlib import Path

import numpy as np
from bcipy.signal.model import SignalModel
from sklearn.mixture import GaussianMixture
from bcipy.helpers.stimuli import GazeReshaper
from sklearn.model_selection import cross_val_score  # noqa
from sklearn.utils.estimator_checks import check_estimator  # noqa
import scipy.stats as stats
from typing import Tuple


class GazeModelIndividual(SignalModel):
    reshaper = GazeReshaper()

    def __init__(self, num_components=2):
        self.num_components = num_components   # number of gaussians to fit

    def fit(self, train_data: np.ndarray) -> 'GazeModelIndividual':
        model = GaussianMixture(n_components=1, random_state=0, init_params='kmeans')
        model.fit(train_data)
        self.model = model

        return self

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compute mean and covariance of each mixture component.
        '''

        means = self.model.means_
        covs = self.model.covariances_

        return means, covs

    def predict(self, test_data: np.ndarray, means: np.ndarray, covs: np.ndarray):
    # def predict(self, test_data: np.ndarray):
        '''
        Compute log-likelihood of each sample.
        Predict the labels for the test data.
        '''
        # TODO: Do we select the means & covs subset of letters in the inquiry or the entire alphabet?
        # means, covs = self.evaluate()  #TODO
        means = np.squeeze(np.array(means))
        covs = np.squeeze(np.array(covs))
        num_components = len(means)

        data_length, _ = test_data.shape

        likelihoods = np.zeros((data_length, num_components), dtype=object)
        predictions = np.zeros(data_length, dtype=object)

        # Find the likelihoods by insterting the test data into the pdf of each component
        for i in range(data_length):
            for k in range(num_components):
                mu = means[k]
                sigma = covs[k]

                likelihoods[i, k] = stats.multivariate_normal.pdf(test_data[i], mu, sigma)

            # Find the argmax of the likelihoods to get the predictions
            predictions[i] = np.argmax(likelihoods[i])

        return likelihoods, predictions

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

    def evaluate(self, sym_pos: np.ndarray):
        '''
        Return mean and covariance of each mixture component.
        All components have the same mean (aligned with symbol position) and covariance.

        test_data: gaze data for each symbol
        sym_pos: mid positions for each symbol in Tobii coordinates
        '''

        means = self.model.means_ + sym_pos
        covs = self.model.covariances_

        return means, covs

    def predict(self, test_data: np.ndarray, means: np.ndarray, covs: np.ndarray):
        '''
        Compute log-likelihood of each sample.
        Predict the labels for the test data.
        '''
        # Compute over log-likelihood scores
        # Get the argmax of the scores

        num_components = len(means)

        data_length, _ = test_data.shape

        likelihoods = np.zeros((data_length, num_components), dtype=object)
        predictions = np.zeros(data_length, dtype=object)

        # Find the likelihoods by insterting the test data into the pdf of each component
        for i in range(data_length):
            for k in range(num_components):
                mu = means[k]
                sigma = covs[k]

                likelihoods[i, k] = stats.multivariate_normal.pdf(test_data[i], mu, sigma)

            # Find the argmax of the likelihoods to get the predictions
            predictions[i] = np.argmax(likelihoods[i])

        return likelihoods, predictions

    def calculate_acc(self, predictions: int, counter: int):
        '''
        Compute model performance characteristics on the provided test data and labels.

        predictions: predicted labels for each test point per symbol
        counter: true labels for each test point per symbol
        '''
        accuracy_per_symbol = np.sum(predictions == counter) / len(predictions) * 100

        return accuracy_per_symbol

    def save(self, path: Path):
        """Save model state to the provided checkpoint"""
        ...

    def load(self, path: Path):
        """Load model state from the provided checkpoint"""
        ...
