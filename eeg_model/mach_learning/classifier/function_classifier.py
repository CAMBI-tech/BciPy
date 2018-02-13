import numpy as np
from eeg_model.mach_learning.m_estimator.m_estimator import *


class RegularizedDiscriminantAnalysis:
    """ Regularized Discriminant Analysis for quadratic boundary in high
    dimensional spaces. Fits discriminant function,
        gi(x) = ln(pi(x)) - (1/2)(x-m)T E^(-1)(x-m) - ln(2pi|E|)
    uses -gi(x) as negative log probability for classification
    Ref:
        Friedman, Jerome H. "Regularized discriminant analysis."
        Journal of the American statistical association 84.405 (1989): 165-175
    Attr:
        lam(float): shrinkage param
        gam(float): threshold param (a.k.a. regularization param)
        class_i(list[int]): class labels
        mean_i(list[ndarray]): list of k x 1 dimensional mean vectors
        prior_i(list[float]): list of prior probabilities
        k (int): Number of features in one sample
        S (ndarray): k x k ndarray
            total data covariance matrix multiplied by number of data samples N
        N (int): Number of samples in x
        cov (ndarray): k x k ndarray, sample covariance of x
        S_i (list[ndarray]): k x k ndarray
            class covariance matrix multiplied by number of data samples N_i in class i
        N_i (list[int]): Number of samples in class i
        cov_i (list[ndarray]): list of k x k ndarray, sample covariance for class i
        log_det_cov(list[float]): list of negative det(cov_i)
        inv_reg_cov_i(list[ndarray]): inverse of regularized covariance matrix for class i

    """

    def __init__(self): # TODO: Make it more modular
        self.lam = .9
        self.gam = .1

        self.class_i = None
        self.mean_i = None
        self.prior_i = None

        self.k = None

        self.S = None
        self.N = None

        self.S_i = None
        self.N_i = None
        self.cov_i = None

        self.log_det_reg_cov_i = None
        self.inv_reg_cov_i = None

    def fit(self, x, y, p=[]):
        """ Fits mean and covariance to the provided data
            and computes regularized covariances based on hyper parameters
            Args:
                x(ndarray[float]): N x k data array
                    N is number of samples k is dimensionality of features
                y(ndarray[int]): N x 1 observation (class) array
                p(ndarray[float]): c x 1 array with prior probabilities
                    c is number of classes in data
                """

        self.N, self.k = x.shape

        # Unique labels/classes
        self.class_i = np.unique(y)

        # Number of data samples in each class
        self.N_i = [np.sum(y == i)
                    for i in self.class_i]

        # MATLAB gets confused if np.where is not used. Insert this relation
        #  in order to make the ndarray readable from MATLAB side. There are
        #  two arrays, [0] for the correctness, choose it
        # Class means
        self.mean_i = [np.mean(x[np.where(y == i)[0]], axis=0)
                       for i in self.class_i]

        # Normalized x
        norm_vec = [x[np.where(y == self.class_i[i])[0]] - self.mean_i[i]
                    for i in range(len(self.class_i))]

        # Outer product of data matrix, Xi'Xi for each class
        self.S_i = [np.dot(np.transpose(norm_vec[i]), norm_vec[i])
                    for i in range(len(self.class_i))]

        # Sample covariances are calculated Si/Ni for each class
        self.cov_i = [self.S_i[i] / self.N_i[i]
                      for i in range(len(self.class_i))]

        # Sample covariance of total data
        self.S = np.zeros((self.k, self.k))

        for i in range(len(self.class_i)):
            self.S += self.S_i[i]

        # Set priors
        if len(p) == 0:
            prior = np.asarray([np.sum(y == self.class_i[i]) for i in
                                range(len(self.class_i))], dtype=float)
            self.prior_i = np.divide(prior, np.sum(prior))
        else:
            self.prior_i = p

        self.regularize(param=[self.gam, self.lam])

    def regularize(self, param): # TODO: what if no param passed?
        """ Regularizes the covariance based on hyper parameters
            Args:
                param(list[gam(float),lam(float)]): List of regularization
                    parameters. Parameters should be a list instead of
                    individual elements for training purposes.
                 """

        self.lam = param[0]
        self.gam = param[1]

        # Shrinked class covariances
        shr_cov_i = [((1 - self.lam) * self.S_i[i] + self.lam * self.S) /
                     ((1 - self.lam) * self.N_i[i] + self.lam * self.N)
                     for i in range(len(self.class_i))]

        # Regularized class covariances
        reg_cov_i = [((1 - self.gam) * shr_cov_i[i] +
                      self.gam / self.k * np.trace(shr_cov_i[i]) *
                      np.eye(self.k)) for i in range(len(self.class_i))]

        self.inv_reg_cov_i, self.log_det_reg_cov_i = [], []

        # Use QR decomposition to find inverse of regularized covariance matrices
        # and their log of determinants
        for i in range(len(self.class_i)):
            q, r = np.linalg.qr(reg_cov_i[i])
            self.inv_reg_cov_i.append(np.linalg.solve(r, np.transpose(q)))
            # self.log_det_reg_cov_i.append(np.sum(np.log(np.abs(np.diag(r)))))

    def transform(self, x):

        val = self.get_prob(x)
        # as the val includes negative log likelihoods it outputs the
        # likelihood ratio for log(p(x|l=1)/p(x|l=0))
        if val.shape[1] == 2:
            val = val[:, 1] - val[:, 0]

        return val

    def get_prob(self, x):
        """ Gets -log likelihoods for each class
            Args:
                x(ndarray): N x k data array where
                    N is number of samples k is dimensionality of features
            Return:
                neg_log_l(ndarray): N x c negative log likelihood array
                    N is number of samples c is number of classes
                """

        neg_log_l = np.zeros([x.shape[0], len(self.class_i)])
        for s in range(x.shape[0]):
            for i in range(len(self.class_i)):
                zero_mean = x[s] - self.mean_i[i]

                # Every constant at the end of score calculation is omitted.
                # This is why we omit log det of class regularized covariances.
                evidence = np.dot(zero_mean,
                                  np.dot(self.inv_reg_cov_i[i],zero_mean))

                neg_log_l[s][i] = -.5*evidence + np.log(self.prior_i[i])

        return neg_log_l

    def fit_transform(self, x, y, p=[]):
        """ Fits the model to provided (x,y) = (data,obs) couples and
        returns the negative log likelihoods.
            Args:
                x(ndarray[float]): N x k data array
                    N is number of samples k is dimensionality of features
                y(ndarray[int]): N x 1 observation (class) array
                p(ndarray[float]): c x 1 array with prior probabilities
                    c is number  of classes in data
            Return:
                val(ndarray[float]): N x c negative log likelihood array
                """

        self.fit(x, y, p)

        return self.transform(x)


class MDiscriminantAnalysis:
    """
    Attr:
    """

    def __init__(self):
        # means and covariances of each channel with the order inherent in data.
        self.means = None
        self.inv_covariances =  None
        self.labels = None
        self.priors = None
        self.toeplitz_inverse_cov = None

    def fit(self, x, y, p=[]):
        """
        :list x: data, each element is every channel's trials.
        """

        # number of channels
        C = len(x)
        p_new = 0
        for index in range(C):
            p_new += len(x[index][0])

        # first index is channel second index is label
        self.means = [[] for i in range(C)]
        self.inv_covariances = [[] for i in range(C)]
        self.labels = np.unique(y)
        self.toeplitz_inverse_cov = [np.zeros((p_new, p_new)) for i in range(len(self.labels))]

        for label_index in range(len(self.labels)):
            count = 0
            for channel in range(C):
                X = x[channel]
                X_label = X[np.where(y == self.labels[label_index])[0], :]
                mean, sigma = robust_mean_covariance(X_label)
                self.means[channel].append(mean)
                self.inv_covariances[channel].append(np.linalg.inv(sigma))
                p_channel = len(self.inv_covariances[channel][0][0])
                self.toeplitz_inverse_cov[label_index][count:count + p_channel, count:count + p_channel] \
                    = self.inv_covariances[channel][label_index]
                count += p_channel

        # Set priors
        if len(p) == 0:
            prior = np.asarray([np.sum(y == self.labels[i]) for i in
                                range(len(self.labels))], dtype=float)
            self.priors = np.divide(prior, np.sum(prior))
        else:
            self.priors = p

    def transform(self, x):
        N = x[0].shape[0]

        scores =[np.zeros(N) for i in range(len(self.labels))]

        for i in range(len(self.labels)):
            for n in range(N):
                current_score = 0
                for c in range(len(x)):
                    temp = x[c][n] - self.means[c][i]
                    current_score += -.5*np.dot(np.dot(temp, self.inv_covariances[c][i]), temp)
                scores[i][n] = current_score + np.log(self.priors[i])

        return scores[1] - scores[0]

    def fit_transform(self, x, y):

        self.fit(x, y)
        return self.transform(x)