import numpy as np


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
        self.lam = 1.
        self.gam = 1.

        self.classes = None
        self.means = []
        self.cov = []
        self.log_det_cov = []
        self.inverse_cov = []
        self.prior = []

    def fit(self, x, y, p=None):
        """ Fits the model to provided (x,y) = (data,obs) couples
            Args:
                x(ndarray[float]): N x k data array
                y(ndarray[int]): N x k observation (class) array
                    N is number of samples k is dimensionality of features
                p(ndarray[float]): c x 1 array with prior probabilities
                    c is number of classes in data
                """

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
        if not p:
            prior = np.asarray([np.sum(y == self.classes[i]) for i in
                                range(len(self.classes))], dtype=float)
            self.prior = np.divide(prior, np.sum(prior))
        else:
            self.prior = p

        sum_cov = sum(self.cov)
        # Regularize covariances
        reg_prior = ((1 - self.lam) * self.prior * x.shape[0]) + (
            self.lam * x.shape[0])
        reg_cov = [((1 - self.lam) * self.cov[i] + self.lam * sum_cov) / (
            reg_prior[i]) for i in range(len(self.classes))]

        # Shrink and store covariances
        self.inverse_cov, self.log_det_cov = [], []
        for i in range(len(self.classes)):
            shr_cov = (1 - self.gam) * reg_cov[i] + (
                ((self.gam / x.shape[1]) * np.trace(reg_cov[i])) *
                np.eye(x.shape[1]))

            q, r = np.linalg.qr(shr_cov)

            # TODO: problem here in linalg solve
            self.inverse_cov.append(np.linalg.solve(r, np.transpose(q)))
            self.log_det_cov.append(np.sum(np.log(np.abs((np.diag(r))))))

    def update(self, x, y):
        # TODO: Implement update method. Different than fit it does not throw
        #  previously tranied model
        asd = 1

    def get_prob(self, x):
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
                evidence = self.log_det_cov[i] + np.dot(zero_mean, np.dot(
                    self.inverse_cov[i], zero_mean))

                neg_log_l[s][i] = (-evidence / 2) + np.log(self.prior[i])

        return neg_log_l


