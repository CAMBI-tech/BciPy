from pathlib import Path

import numpy as np
from numpy.core.multiarray import array as array
from bcipy.signal.model import SignalModel
from sklearn.mixture import GaussianMixture
from bcipy.helpers.stimuli import GazeReshaper
from sklearn.model_selection import cross_val_score  # noqa
from sklearn.utils.estimator_checks import check_estimator  # noqa
import scipy.stats as stats
from typing import List, Tuple
from numpy.linalg import inv
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow

import matplotlib.pyplot as plt
import numpy as np


# def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
#     X = X.ravel()
#     mu = mu.ravel()
#     uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
#     plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
#     plt.plot(X, mu, label='Mean')
#     for i, sample in enumerate(samples):
#         plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
#     if X_train is not None:
#         plt.plot(X_train, Y_train, 'rx')
#     plt.legend()

# def plot_model(m, X, Y, lower=-8.0, upper=8.0):
#     pX = np.linspace(lower, upper, 100)[:, None]
#     pY, pYv = m.predict_y(pX)
#     if pY.ndim == 3:
#         pY = pY[:, 0, :]
#     plt.plot(X, Y, "x")
#     plt.gca().set_prop_cycle(None)
#     plt.plot(pX, pY)
#     for i in range(pY.shape[1]):
#         top = pY[:, i] + 2.0 * pYv[:, i] ** 0.5
#         bot = pY[:, i] - 2.0 * pYv[:, i] ** 0.5
#         plt.fill_between(pX[:, 0], top, bot, alpha=0.3)
#     plt.xlabel("X")
#     plt.ylabel("f")
#     plt.title(f"ELBO: {m.elbo([X, Y]):.3}")
#     plt.plot(Z, Z * 0.0, "o")

class KernelGP(SignalModel):
    def __init__(self):
        reshaper = GazeReshaper()

    def fit(self, training_data: np.ndarray, training_labels: np.ndarray):
        training_data = np.asarray(training_data)



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

class KernelGPSampleAverage(SignalModel):
    reshaper = GazeReshaper()
    
    def __init__(self):
        self.ready_to_predict = False

    def fit(self, training_data: np.ndarray):
        training_data = np.array(training_data)
        # Training data shape = inquiry x features x samples
        # reshape training data to inquiry x (features x samples)
        reshaped_data = training_data.reshape((len(training_data), -1))
        cov_matrix = np.cov(reshaped_data, rowvar=False)
        # cov_matrix_shape = (features x samples) x (features x samples)
        reshaped_mean = np.mean(reshaped_data, axis=0)
        
        
    
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
            new_data[0:2,i] = data[0:2,i] - symbol_pos
            new_data[2:4,i] = data[2:4,i] - symbol_pos

        return new_data
    
    def substract_mean(self, data: np.ndarray, time_avg: np.ndarray) -> np.ndarray:
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
            new_data[:,i] = data[:,i] - time_avg

        return new_data


class GMIndividual(SignalModel):
    """Gaze model that fits different Gaussians/Gaussian Mixtures for each symbol."""
    reshaper = GazeReshaper()

    def __init__(self, num_components=2, random_state=0):
        self.num_components = num_components   # number of gaussians to fit
        self.random_state = random_state
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

    def evaluate(self, test_data, test_labels) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Evaluate the accuracy of the model.
        '''
        ...
    
    def compute_likelihood_ratio(self, data: np.array, inquiry: List[str], symbol_set: List[str]) -> np.array:
        '''
        Not implemented in this model.
        '''
        ...

    def predict(self, test_data: np.ndarray, inquiry, symbol_set) -> np.ndarray:
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
        # posterior = likelihoods x prior / common denominator

        return likelihoods
        # return posterior
        # TODO: calculate posteriors for each symbol, using the prior information!!

    def evaluate_likelihood(self, data: np.ndarray) -> np.ndarray:
        data_length, _ = data.shape

        likelihoods = np.zeros((data_length, self.num_components), dtype=object)

        # Find the likelihoods by insterting the test data into the pdf of each component
        for i in range(data_length):
            for k in range(self.num_components):
                mu = self.means[k]
                sigma = self.covs[k]

                likelihoods[i, k] = stats.multivariate_normal.pdf(data[i], mu, sigma)

                #likelihoods[0, k] = [() , (), ... ]

                # p(x_g_i | theta = A), p(x_g_i | theta = B), p(x_g_i | theta = C), ...
        # TODO: multiply over all data[i]
        """

        """
        # return log_likelihoods
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

    def __init__(self, num_components=1, random_state=0):
        self.num_components = num_components   # number of gaussians to fit
        self.random_state = random_state
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

    def evaluate(self, test_data, test_labels):
        '''
        Evaluate the accuracy of the model.
        '''

        ...

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        '''
        Compute log-likelihood of each sample.
        Predict the labels for the test data.
        '''
        data_length, _ = test_data.shape
        # TODO what is this data length?
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
        TODO: This could be our evaluation function
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
            new_data[i] = data[i] - symbol_pos

        return new_data