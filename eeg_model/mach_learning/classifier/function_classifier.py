import numpy as np

import numpy as np


class RegularizedDiscriminantAnalysisX(object):
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

    def __init__(self):  # TODO: Make it more modular
        self.lam = .1
        self.gam = .1

        self.class_i = None
        self.mean_i = None
        self.prior_i = None

        self.k = None

        self.S = None
        self.N = None
        self.cov = None

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
        self.cov = np.cov(x, rowvar=False)
        self.S = self.cov * self.N

        # Set prior information on observed data if not given
        if len(p) == 0:
            prior = np.asarray([np.sum(y == self.class_i[i]) for i in
                                range(len(self.class_i))], dtype=float)
            self.prior_i = np.divide(prior, np.sum(prior))
        else:
            self.prior_i = p

        self.regularize(param=[self.gam, self.lam])

    def regularize(self, param):
        """ Regularizes the covariance based on hyper parameters
            Args:
                param(list[gam(float),lam(float)]): List of regularization
                    parameters. Parameters should be a list instead of
                    individual elements for training purposes.
                 """

        # TODO: what if no param passed?
        self.gam = param[0]
        self.lam = param[1]

        # Shrinked class covariances
        shr_cov_i = [((1 - self.lam) * self.S_i[i] + self.lam * self.S) /
                     ((1 - self.lam) * self.N_i[i] + self.lam * self.N)
                     for i in range(len(self.class_i))]

        # Regularized class covariances
        reg_cov_i = [((1 - self.gam) * shr_cov_i[i]
                      + self.gam / self.k * np.trace(shr_cov_i[i]) * np.eye(
            self.k))
                     for i in range(len(self.class_i))]

        self.inv_reg_cov_i, self.log_det_reg_cov_i = [], []

        # Use QR decomposition to find inverse of regularized covariance matrices
        # and their log of determinants
        for i in range(len(self.class_i)):
            q, r = np.linalg.qr(reg_cov_i[i])
            self.inv_reg_cov_i.append(np.linalg.solve(r, np.transpose(q)))
            self.log_det_reg_cov_i.append(np.sum(np.log(np.abs(np.diag(r)))))

    def update(self, x, y):
        # TODO: Implement update method. Different than fit it does not throw
        #  previously trained model
        asd = 1

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
                evidence = self.log_det_reg_cov_i[i] + np.dot(zero_mean,
                                                              np.dot(
                                                                  self.inv_reg_cov_i[
                                                                      i],
                                                                  zero_mean))

                neg_log_l[s][i] = (-evidence / 2) + np.log(self.prior_i[i])

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


class RegularizedDiscriminantAnalysis(object):
    """ Regularized Discriminant Analysis for quadratic boundary in high
    dimensional spaces. fits discriminant function,
        gi(x) = ln(pi(x)) - (1/2)(x-m)T E^(-1)(x-m) - ln(2pi|E|)
    uses -gi(x) as negative log probability for classification
    Ref:
        Friedman, Jerome H. "Regularized discriminant analysis."
        Journal of the American statistical association 84.405 (1989): 165-175
    Attr:
        lam(float):shrinkage param
        gam(float): threshold  param
        classes(list[int]): class labels with
        means(list[ndarray]): list of k x 1 dimensional mean vectors
        cov(list[ndarray]): list of N x k dimensional covariance arrays
        log_det_cov(list[float]): list of negative det(cov)
        inverse_cov(list[ndarray]): list of N x k inverse covariance arrays
        prior(list[float]): list of prior probabilities
    """

    def __init__(self):
        self.lam = .9
        self.gam = .1

        self.classes = None
        self.means = []
        self.cov = []
        self.reg_log_det_cov = []
        self.reg_inverse_cov = []
        self.prior = []

        self.N = 1
        self.k = 1

    # TODO: Make it more modular

    def fit(self, x, y, p=[]):
        """ Fits mean and covariance to the provided data
            and computes regularized covariances based on hyper parameters
            Args:
                x(ndarray[float]): N x k data array
                y(ndarray[int]): N x 1 observation (class) array
                    N is number of samples k is dimensionality of features
                p(ndarray[float]): c x 1 array with prior probabilities
                    c is number of classes in data
                """

        self.N = x.shape[0]
        self.k = x.shape[1]
        self.classes = np.unique(y)
        # MATLAB gets confused if np.where is not used. Insert this relation
        #  in order to make the ndarray readable from MATLAB side. There are
        #  two arrays, [0] for the correctness, choose it
        self.means = [np.mean(x[np.where(y == self.classes[i])[0]], axis=0)
                      for i in range(len(self.classes))]
        norm_vec = [x[np.where(y == self.classes[i])[0]] - self.means[i] for
                    i in range(len(self.classes))]

        # self.cov = [np.cov(np.transpose(x[np.where(y == self.classes[i])[0]]))
        #             for i in range(len(self.classes))]

        self.cov = [np.dot(np.transpose(norm_vec[i]), norm_vec[i]) for
                    i in range(len(self.classes))]

        # Set prior information on observed data if not given
        if not len(p):
            prior = np.asarray([np.sum(y == self.classes[i]) for i in
                                range(len(self.classes))], dtype=float)
            self.prior = np.divide(prior, np.sum(prior))
        else:
            self.prior = p

        self.regularize(param=[self.gam, self.lam])

    def regularize(self, param):
        """ Regularizes the covariance based on hyper parameters
            Args:
                param(list[gam(float),lam(float)]): List of regularization
                    parameters. Parameters should be a list instead of
                    individual elements for training purposes.
                 """

        # TODO: what if no param passed?
        self.gam = param[0]
        self.lam = param[1]

        sum_cov = sum(self.cov)
        # Regularize covariances
        # TODO: prior solution!
        reg_prior = ((1 - self.lam) * self.prior * self.N) + \
                    (self.lam * self.N)
        reg_cov = [((1 - self.lam) * self.cov[i] + self.lam * sum_cov) / (
            reg_prior[i]) for i in range(len(self.classes))]

        # Shrink and store covariances
        self.reg_inverse_cov, self.reg_log_det_cov = [], []
        for i in range(len(self.classes)):
            shr_cov = (1 - self.gam) * reg_cov[i] + (
                ((self.gam / self.k) * np.trace(reg_cov[i])) * np.eye(self.k))

            q, r = np.linalg.qr(shr_cov)

            self.reg_inverse_cov.append(np.linalg.solve(r, np.transpose(q)))
            self.reg_log_det_cov.append(np.sum(np.log(np.abs((np.diag(r))))))

    def update(self, x, y):
        # TODO: Implement update method. Different than fit it does not throw
        #  previously tranied model
        asd = 1

    def transform(self, x):
        val = self.get_proba(x)
        # as the val includes negative log likelihoods it outputs the
        # likelihood ratio for log(p(x|l=1)/p(x|l=0))
        if val.shape[1] == 2:
            val = val[:, 1] - val[:, 0]

        return val

    def get_proba(self, x):
        """ Gets -log likelihoods for each class
            Args:
                x(ndarray): N x k data array where
                    N is number of samples k is dimensionality of features
            Return:
                neg_log_l(ndarray): N x c negative log likelihood array
                    N is number of samples c is number of classes
                """

        neg_log_l = np.zeros([x.shape[0], len(self.classes)])
        for s in range(x.shape[0]):
            for i in range(len(self.classes)):
                zero_mean = x[s] - self.means[i]
                evidence = self.reg_log_det_cov[i] + np.dot(zero_mean, np.dot(
                    self.reg_inverse_cov[i], zero_mean)) / 2

                neg_log_l[s][i] = -evidence + np.log(self.prior[i])

        return neg_log_l

    def fit_transform(self, x, y, p=[]):
        """ Fits the model to provided (x,y) = (data,obs) couples and
        returns the negative log likelihoods.
            Args:
                x(ndarray[float]): N x k data array
                y(ndarray[int]): N x k observation (class) array
                    N is number of samples k is dimensionality of features
                p(ndarray[float]): c x 1 array with prior probabilities
                    c is number  of classes in data
            Return:
                val(ndarray[float]): N x c negative log likelihood array
                """

        self.fit(x, y, p)
        val = self.transform(x)
        return val

    def predict(self, x):
        tmp = self.transform(x)
        arg = (np.exp(tmp) >= .5) + 0

        return arg
