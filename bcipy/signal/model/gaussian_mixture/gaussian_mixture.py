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


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()

class GazeModelKernelGaussianProcess(SignalModel):
    def __init__(self):
        self.ready_to_predict = False

    def fit(self, training_data: np.ndarray, training_labels: np.ndarray):
        return super().fit(training_data, training_labels)
    
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

class GazeModelKernelGaussianProcessSampleAverage(SignalModel):
    reshaper = GazeReshaper()
    window_length = 3
    
    def __init__(self):
        self.ready_to_predict = False

    def fit(self, training_data: np.ndarray):
        # Multivariate Gaussian Process Solver

        # How to det the kernel function? The current one is an RBF kernel
        def mykernel(x1,x2):
            a = 1
            l = 2
            return a*np.exp(-((x1-x2)**2)/(2*l**2))
        
        # Another kernel function
        def mykernel2(x1,x2):
            H = 0.3
            return np.abs(x1)**(2*H) + np.abs(x2)**(2*H) - np.abs(x1-x2)**(2*H)
        
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
        
        # K = mykernel(np.reshape(training_data,(1,-1)), np.reshape(training_data,(1,-1)))

        # # visualization of kernel matrix
        # fig = plt.figure(figsize=(10,10)) # scale of the plot
        # image = plt.imshow(K)
        # plt.show()

        # mju = np.zeros(len(training_data)) # mean vector

        # R = np.random.multivariate_normal(mju,K)

        # fig2 = plt.figure(figsize=(10,5)) # scale of the plot
        # plt.plot(training_data, R,'bx')
        # plt.show()

        # # conditions:
        # f_cond = [1, 1]
        # x_cond = [40, 80]
        # mju_cond = [0, 0]

        # K_x_c = np.transpose(mykernel(training_data.reshape((1,-1)),np.transpose(np.asarray(x_cond).reshape((1,-1)))))

        # invK_c_c = inv(mykernel(np.asarray(x_cond).reshape((1,-1)),np.transpose(np.asarray(x_cond).reshape((1,-1)))))

        # mju = np.zeros(d)
        # K = mykernel(training_data.reshape((1,-1)),np.transpose(training_data.reshape((1,-1))))

        # MU = mju + np.matmul(np.matmul(K_x_c,invK_c_c),np.array(f_cond) - np.array(mju_cond))
        # SIGMA = K - np.matmul(np.matmul(K_x_c,invK_c_c),K_x_c.transpose())

        # # m realisations of multivariate conditional Gaussian distribution:
        # RC = np.random.multivariate_normal(MU,SIGMA,m)

        # m = 1 # number of realisations
        # R = np.random.multivariate_normal(mju,K,m)

        # fig = plt.figure(figsize=(15,6))
        # plt.figure(figsize=(15,7))
        # plt.plot((np.ones((m,d)) * training_data).transpose(), RC.transpose(), 'x')
        # plt.show()
        
        # extract list to np array:
        training_data = np.asarray(training_data)

        from numpy.linalg import inv

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
        X_s = training_data[71][:,0]

        X = np.array(range(len(X_train))).reshape(-1, 1)
        # Mean and covariance of the prior
        mu = np.zeros(X.shape)
        cov = kernel(X, X)  

        samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

        # Plot GP mean, uncertainty region and samples 
        plot_gp(mu, cov, X, samples=samples) 

        # Prediction from noise-free training data
        l=1.0 
        sigma_f=1.0
        sigma_y=1e-4  

        K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
        K_s = kernel(X_train, X_s, l, sigma_f)
        K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
        K_inv = inv(K)
        
        # K = mykernel(X_train, X_train) + sigma_y**2 * np.eye(len(X_train))
        # K_s = mykernel(X_train, X_s)
        # K_ss = mykernel(X_s, X_s) + 1e-4 * np.eye(len(X_s))
        # K_inv = inv(K)
        
        # Equation (7)
        mu_s = K_s.T.dot(K_inv).dot(Y_train)

        # Equation (8)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
        plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
        breakpoint()
    
    
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
        for i in range(len(data)):
            new_data[i] = data[i] - symbol_pos

        return new_data


class GazeModelIndividual(SignalModel):
    reshaper = GazeReshaper()
    window_length = 3

    def __init__(self, num_components=2, random_state=0):
        self.num_components = num_components   # number of gaussians to fit
        self.random_state = random_state
        self.means = None
        self.covs = None

        self.ready_to_predict = False

    def fit(self, train_data: np.ndarray) -> 'GazeModelIndividual':
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


class GazeModelCombined(SignalModel):
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