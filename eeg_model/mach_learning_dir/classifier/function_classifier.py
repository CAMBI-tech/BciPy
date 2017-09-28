import numpy as np
import scipy.optimize
from sklearn import metrics


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

    # TODO: Make it more modular

    def fit(self, x, y, p=[], op_type='cost_auc'):
        # TODO: SHOULD WE HAVE A TRAIN TEST SPLIT HERE?
        """ Fits the model to provided (x,y) = (data,obs) couples
            Args:
                x(ndarray[float]): N x k data array
                y(ndarray[int]): N x k observation (class) array
                    N is number of samples k is dimensionality of features
                p(ndarray[float]): c x 1 array with prior probabilities
                    c is number of classes in data
                """
        self.fit_param(x, y, p)
        self.opt_param(x, y, op_type=op_type)
        self.fit_param(x, y, p)

    def fit_param(self, x, y, p=[]):
        """ Fits mean and covariance to the provided data
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
        if not len(p):
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

            self.inverse_cov.append(np.linalg.solve(r, np.transpose(q)))
            self.log_det_cov.append(np.sum(np.log(np.abs((np.diag(r))))))

    def update(self, x, y):
        # TODO: Implement update method. Different than fit it does not throw
        #  previously tranied model
        asd = 1

    def transform(self, x):
        val = self.get_proba(x)
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
                evidence = self.log_det_cov[i] + np.dot(zero_mean, np.dot(
                    self.inverse_cov[i], zero_mean))

                neg_log_l[s][i] = (-evidence / 2) + np.log(self.prior[i])

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
                neg_log_l(ndarray[float]): N x c negative log likelihood array
                """

        self.fit(x, y, p)
        neg_log_l = self.get_proba(x)
        return neg_log_l

    def opt_param(self, x, y, init=None, op_type='cost_auc'):
        """ Optimizes lambda, gamma values for given  penalty function
        Args:
            x(ndarray[float]): N x k data array
            y(ndarray[int]): N x k observation (class) array
                 N is number of samples k is dimensionality of features
            init(list[float]): initial values for gamma and lambda
            op_type(string): type of the optimization
            """

        # Get initial values
        if not init:
            init = [self.lam, self.gam]
        if op_type:
            # TODO: maybe we should not have such an option and set it by ourselves
            if op_type == 'cost_auc':
                cost_fun_param = lambda b: self.cost_auc(x, y, b[0], b[1])

            # Intervals for lambda and gamma parameters
            # Observe that 0 < lam < 1, 0 < gam < 1
            cst_1 = lambda v: v[0]
            cst_2 = lambda v: v[1]
            cst_3 = lambda v: 1 - v[0]
            cst_4 = lambda v: 1 - v[1]

            arg_opt = scipy.optimize.fmin_cobyla(cost_fun_param, x0=init,
                                                 disp=False,
                                                 cons=[cst_1, cst_2, cst_3,
                                                       cst_4])
            self.lam = arg_opt[0]
            self.gam = arg_opt[1]

    # TODO: Insert cost functions for parameter update below!
    def cost_auc(self, x, y, lam, gam):
        """ Minimize cost of the overall -AUC
            Args:
                x(ndarray[float]): N x k data array
                y(ndarray[int]): N x k observation (class) array
                    N is number of samples k is dimensionality of features
                lam(float): cost function lambda to iterate over
                gam(float): cost function gamma to iterate over
            Return:
                -auc(float): negative AUC value for current setup
                """

        # x1, x2, y1, y2 = train_test_split(x, y, test_size=0.1)
        self.lam = lam
        self.gam = gam
        # self.fit_param(x1, y1, self.prior)
        self.fit_param(x, y, self.prior)
        # sc = self.get_proba(x2)
        sc = self.get_proba(x)
        sc = np.dot(np.array([-1, 1]), sc.transpose())
        # fpr, tpr, _ = metrics.roc_curve(y2, sc, pos_label=1)
        fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        return -auc
