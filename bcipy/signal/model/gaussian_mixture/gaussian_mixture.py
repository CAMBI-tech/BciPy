import pickle
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

class GP(SignalModel):
    def __init__(self):
        reshaper = GazeReshaper()
        self.ready_to_predict = False

    def fit(self, training_data: np.ndarray, training_labels: np.ndarray):
        training_data = np.asarray(training_data)
        # GPflow:
        N = training_data.shape[0]
        D = training_data.shape[1]
        M = 15  # number of inducing points
        L = 2  # number of latent GPs
        P = 3  # number of observations = output dimensions


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

class GPSampleAverage(SignalModel):
    reshaper = GazeReshaper()
    window_length = 3
    
    def __init__(self):
        self.ready_to_predict = False

    def fit(self, training_data: np.ndarray):
        # Multivariate Gaussian Process Solver

        # # RBF kernel
        # def mykernel(x1,x2):
        #     a = 1
        #     l = 2
        #     return a*np.exp(-((x1-x2)**2)/(2*l**2))
        
        # # Another kernel function
        # def mykernel2(x1,x2):
        #     H = 0.3
        #     return np.abs(x1)**(2*H) + np.abs(x2)**(2*H) - np.abs(x1-x2)**(2*H)
        
        def kernel(X1, X2, l=1.0, sigma_f=1.0):
            """
            Isotropic squared exponential kernel.
            
            Args:
                X1: Array of m points (m x d).
                X2: Array of n points (n x d).

            Returns:
                (m x n) matrix.
            """
            sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
            return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
        
        # # extract list to np array:
        # training_data = np.asarray(training_data)

        """
        Computes the suffifient statistics of the posterior distribution 
        from m training data X_train and Y_train and n new inputs X_s.
        
        Args:
            X_s: New input locations (n x d).
            X_train: Training locations (m x d).
            Y_train: Training targets (m x 1).
            l: Kernel length parameter.
            sigma_f: Kernel vertical variation parameter.
            sigma_y: Noise parameter.
        
        Returns:
            Posterior mean vector (n x d) and covariance matrix (n x n).
        """
        X_train = training_data[0][:,0].reshape(-1, 1)
        Y_train = np.array(range(len(X_train))).reshape(-1, 1)
        # X_s = training_data[71][:,0]

        # X = np.array(range(len(X_train))).reshape(-1, 1)
        # # Mean and covariance of the prior
        # mu = np.zeros(X.shape)
        # cov = kernel(X, X)  

        # samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

        # Plot GP mean, uncertainty region and samples 
        # plot_gp(mu, cov, X, samples=samples) 

        # Prediction from noise-free training data 
    
    
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
    window_length = 3

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
    
    # def compute_likelihood_ratio(self, data: np.array, inquiry: List[str], symbol_set: List[str]) -> np.array:
    #     '''
    #     Not implemented in this model.
    #     '''
    #     ...

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
        
        '''
        p(g| theta) = p(g| theta = A), p(g| theta = B), p(g| theta = C), ... len(symbol_set)
        '''
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
        
    
    def save(self, path: Path) -> None:
        """Save model weights (e.g. after training) to `path`"""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path) -> SignalModel:
        """Load pretrained model from `path`"""
        with open(path, "rb") as f:
            model = pickle.load(f)

        return model


class GMCentralized(SignalModel):
    '''Gaze model that uses all symbols to fit a single Gaussian '''
    reshaper = GazeReshaper()
    window_length = 3

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