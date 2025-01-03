from pathlib import Path
from typing import List
from enum import Enum

from bcipy.helpers.stimuli import GazeReshaper
from bcipy.signal.model import SignalModel

from sklearn.mixture import GaussianMixture
import numpy as np
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow


class GazeModelType(Enum):
    """Enum for gaze model types"""
    GAUSSIAN_PROCESS = "GaussianProcess"
    GM_INDIVIDUAL = "GMIndividual"
    GM_CENTRALIZED = "GMCentralized"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @staticmethod
    def from_str(label: str):
        if label == "GaussianProcess":
            return GazeModelType.GAUSSIAN_PROCESS
        elif label == "GMIndividual":
            return GazeModelType.GM_INDIVIDUAL
        elif label == "GMCentralized":
            return GazeModelType.GM_CENTRALIZED
        else:
            raise ValueError(f"Model type {label} not recognized.")


class GazeModelResolver:
    """Factory class for gaze models

    This class is responsible for loading gaze models via type resolution.
    """

    @staticmethod
    def resolve(model_type: str, *args, **kwargs) -> SignalModel:
        """Load a gaze model from the provided path."""
        model_type = GazeModelType.from_str(model_type)
        if model_type == GazeModelType.GAUSSIAN_PROCESS:
            return GaussianProcess(*args, **kwargs)
        elif model_type == GazeModelType.GM_INDIVIDUAL:
            return GMIndividual(*args, **kwargs)
        elif model_type == GazeModelType.GM_CENTRALIZED:
            return GMCentralized(*args, **kwargs)
        else:
            raise ValueError(
                f"Model type {model_type} not able to resolve. Not registered in GazeModelResolver.")


class GaussianProcess(SignalModel):

    name = "GaussianProcessGazeModel"
    reshaper = GazeReshaper()

    def __init__(self, *args, **kwargs):
        self.ready_to_predict = False
        self.acc = None

    def fit(self, training_data: np.ndarray):
        ...

    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray):
        ...

    def predict(self, test_data: np.ndarray, inquiry, symbol_set) -> np.ndarray:
        ...

    def predict_proba(self, test_data: np.ndarray) -> np.ndarray:
        ...

    def save(self, path: Path):
        ...

    def load(self, path: Path):
        ...

    def centralize(self, data: np.ndarray, symbol_pos: np.ndarray) -> np.ndarray:
        """ Using the symbol locations in matrix, centralize all data (in Tobii units).
        This data will only be used in certain model types.
        Args:
            data (np.ndarray): Data in shape of num_samples x num_dimensions
            symbol_pos (np.ndarray(float)): Array of the current symbol posiiton in Tobii units
        Returns:
            new_data (np.ndarray): Centralized data in shape of num_samples x num_dimensions
        """
        new_data = np.copy(data)
        for i in range(len(data.T)):
            new_data[0:2, i] = data[0:2, i] - symbol_pos
            new_data[2:4, i] = data[2:4, i] - symbol_pos

        return new_data

    def subtract_mean(self, data: np.ndarray, time_avg: np.ndarray) -> np.ndarray:
        """ Using the symbol locations in matrix, centralize all data (in Tobii units).
        This data will only be used in certain model types.
        Args:
            data (np.ndarray): Data in shape of num_samples x num_dimensions
            symbol_pos (np.ndarray(float)): Array of the current symbol posiiton in Tobii units
        Returns:
            new_data (np.ndarray): Centralized data in shape of num_samples x num_dimensions
        """
        new_data = np.copy(data)
        for i in range(len(data.T)):
            new_data[:, i] = data[:, i] - time_avg

        return new_data


class GMIndividual(SignalModel):
    """Gaze model that fits different Gaussians/Gaussian Mixtures for each symbol."""
    reshaper = GazeReshaper()
    name = "gaze_model_individual"

    def __init__(self, num_components=4, random_state=0, *args, **kwargs):
        self.num_components = num_components   # number of gaussians to fit
        self.random_state = random_state
        self.acc = None
        self.means = None
        self.covs = None

        self.ready_to_predict = False

    def fit(self, train_data: np.ndarray):
        model = GaussianMixture(n_components=self.num_components, random_state=self.random_state, init_params='kmeans')
        model.fit(train_data)
        self.model = model

        self.means = model.means_
        self.covs = model.covariances_

        self.ready_to_predict = True

        return self

    def evaluate(self, predictions, true_labels) -> np.ndarray:
        '''
        Compute performance characteristics on the provided test data and labels.

        Parameters:
        -----------
        predictions: predicted labels for each test point per symbol
        true_labels: true labels for each test point per symbol
        Returns:
        --------
        accuracy_per_symbol: accuracy per symbol
        '''
        accuracy_per_symbol = np.sum(predictions == true_labels) / len(predictions) * 100
        self.acc = accuracy_per_symbol
        return accuracy_per_symbol

    def compute_likelihood_ratio(self, data: np.array, inquiry: List[str], symbol_set: List[str]) -> np.array:
        '''
        Not implemented in this model.
        '''
        ...

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        '''
        Compute log-likelihood of each sample.
        Predict the labels for the test data.

        test_data: gaze data
        inquiry: the subset of symbols presented
        symbol_set: the entire set of symbols presented
        '''
        data_length, _ = test_data.shape
        predictions = np.zeros(data_length, dtype=object)

        likelihoods = self.model.predict_proba(test_data)

        for i in range(data_length):
            # Find the argmax of the likelihoods to get the predictions
            predictions[i] = np.argmax(likelihoods[i])

        return predictions

    def predict_proba(self, test_data: np.ndarray) -> np.ndarray:
        '''
        Compute log-likelihood of each sample.

        test_data: gaze data

        log(p(l | gaze)) = log(p(gaze | l)) + log(p(l)) - log(p(gaze))
        '''
        data_length, _ = test_data.shape

        likelihoods = np.zeros((data_length, self.num_components), dtype=object)

        # Find the likelihoods by insterting the test data into the pdf of each component
        for i in range(data_length):
            for k in range(self.num_components):
                mu = self.means[k]
                sigma = self.covs[k]

                likelihoods[i, k] = stats.multivariate_normal.pdf(test_data[i], mu, sigma)

        return likelihoods

    def evaluate_likelihood(self, data: np.ndarray) -> np.ndarray:
        data_length, _ = data.shape

        likelihoods = np.zeros((data_length, self.num_components), dtype=object)

        # Find the likelihoods by insterting the test data into the pdf of each component
        for i in range(data_length):
            for k in range(self.num_components):
                mu = self.means[k]
                sigma = self.covs[k]

                likelihoods[i, k] = stats.multivariate_normal.pdf(data[i], mu, sigma)

        return likelihoods

    def save(self, path: Path):
        """Save model state to the provided checkpoint"""
        ...

    def load(self, path: Path):
        """Load model state from the provided checkpoint"""
        ...


class GMCentralized(SignalModel):
    '''Gaze model that uses all symbols to fit a single Gaussian '''
    reshaper = GazeReshaper()
    name = "gaze_model_combined"

    def __init__(self, num_components=4, random_state=0, *args, **kwargs):
        self.num_components = num_components   # number of gaussians to fit
        self.random_state = random_state
        self.acc = None
        self.means = None
        self.covs = None

        self.ready_to_predict = False

    def fit(self, train_data: np.ndarray):
        model = GaussianMixture(n_components=self.num_components, random_state=self.random_state, init_params='kmeans')
        model.fit(train_data)
        self.model = model

        self.means = model.means_
        self.covs = model.covariances_

        self.ready_to_predict = True
        return self

    def evaluate(self, predictions, true_labels) -> np.ndarray:
        '''
        Compute performance characteristics on the provided test data and labels.

        Parameters:
        -----------
        predictions: predicted labels for each test point per symbol
        true_labels: true labels for each test point per symbol
        Returns:
        --------
        accuracy_per_symbol: accuracy per symbol
        '''
        accuracy_per_symbol = np.sum(predictions == true_labels) / len(predictions) * 100
        self.acc = accuracy_per_symbol
        return accuracy_per_symbol

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        '''
        Compute log-likelihood of each sample.
        Predict the labels for the test data.
        '''
        data_length, _ = test_data.shape
        predictions = np.zeros(data_length, dtype=object)
        likelihoods = self.model.predict_proba(test_data)

        for i in range(data_length):
            # Find the argmax of the likelihoods to get the predictions
            predictions[i] = np.argmax(likelihoods[i])

        return predictions

    def predict_proba(self, test_data: np.ndarray) -> np.ndarray:
        '''
        Compute log-likelihood of each sample.
        Predict the labels for the test data.

        test_data:
        '''
        data_length, _ = test_data.shape

        likelihoods = np.zeros((data_length, self.num_components), dtype=object)

        # Find the likelihoods by insterting the test data into the pdf of each component
        for i in range(data_length):
            for k in range(self.num_components):
                mu = self.means[k]
                sigma = self.covs[k]

                likelihoods[i, k] = stats.multivariate_normal.pdf(test_data[i], mu, sigma)

        return likelihoods

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

    def centralize(self, data: np.ndarray, symbol_pos: np.ndarray) -> np.ndarray:
        """ Using the symbol locations in matrix, centralize all data (in Tobii units).
        This data will only be used in certain model types.
        Args:
            data (np.ndarray): Data in shape of num_samples x num_dimensions
            symbol_pos (np.ndarray(float)): Array of the current symbol posiiton in Tobii units
        Returns:
            new_data (np.ndarray): Centralized data in shape of num_samples x num_dimensions
        """
        new_data = np.copy(data)
        for i in range(len(data)):
            # new_data[i] = data[i] - symbol_pos
            new_data[:2, i] = data[:2, i] - symbol_pos
            new_data[2:, i] = data[2:, i] - symbol_pos

        return new_data
