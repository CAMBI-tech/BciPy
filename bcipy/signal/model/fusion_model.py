from pathlib import Path

import numpy as np
from bcipy.signal.model import SignalModel
from sklearn.mixture import GaussianMixture
from bcipy.helpers.stimuli import GazeReshaper
from sklearn.model_selection import cross_val_score  # noqa
from sklearn.utils.estimator_checks import check_estimator  # noqa


class GazeModel(SignalModel):
    reshaper = GazeReshaper()

    def __init__(self, num_components=2):
        self.num_components = num_components   # number of gaussians to fit

    def fit(self, train_data: np.array):
        model = GaussianMixture(n_components=2, random_state=0, init_params='kmeans')
        model.fit(train_data)
        self.model = model

        return self

    def get_scores(self, test_data: np.array):
        '''
        Compute the log-likelihood of each sample.
        Compute the mean and covariance of each mixture component.
        '''

        scores = self.model.score_samples(test_data)
        means = self.model.means_
        covs = self.model.covariances_

        return scores, means, covs

    def predict(self, scores: np.array):
        '''
        Predict the labels for the test data.
        '''
        # Compute over log-likelihood scores
        # Get the argmax of the scores

        # return predictions

    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray):
        '''
        Compute model performance characteristics on the provided test data and labels.
        '''
        # Compute the AUC

        # return ModelEvaluationReport(auc)

    def save(self, path: Path):
        """Save model state to the provided checkpoint"""
        ...

    def load(self, path: Path):
        """Load model state from the provided checkpoint"""
        ...


class GazeModel_AllSymbols(SignalModel):
    '''Gaze model that uses all symbols to fit a single Gaussian '''
    reshaper = GazeReshaper()

    def __init__(self, num_components=1):
        self.num_components = num_components   # number of gaussians to fit

    def fit(self, train_data: np.array):
        model = GaussianMixture(n_components=1, random_state=0, init_params='kmeans')
        model.fit(train_data)
        self.model = model

        return self

    def get_scores(self, test_data: np.array, sym_pos: np.array):
        '''
        Compute the log-likelihood of each sample.
        Return the mean and covariance of each mixture component.

        test_data: gaze data for each symbol
        sym_pos: mid positions for each symbol in Tobii coordinates
        '''

        scores = self.model.score_samples(test_data)
        means = self.model.means_ + sym_pos
        covs = self.model.covariances_

        return scores, means, covs

    def predict(self, scores: np.array):
        '''
        Predict the labels for the test data.
        '''
        # Compute over log-likelihood scores
        # Get the argmax of the scores

        # return predictions

    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray):
        '''
        Compute model performance characteristics on the provided test data and labels.
        '''
        # Compute the AUC

        # return ModelEvaluationReport(auc)

    def save(self, path: Path):
        """Save model state to the provided checkpoint"""
        ...

    def load(self, path: Path):
        """Load model state from the provided checkpoint"""
        ...
