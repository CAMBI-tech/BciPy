from sklearn.neighbors.kde import KernelDensity
import numpy as np


class KernelDensityEstimate(object):
    """ Kernel density estimate implementation using scikit learn
    library. For further reference, please check scikit learn website.
    Attr:
        bandwidth(float): bandwidth of the kernel
        algorithm(string): algorithm type
        kernel(string): element toform the actual fitted pdf.
        metric(string): distance metric used by algorithm to insert kernels
            onto samples in the given domain.
        atol(float): absolute tolerance to fit a probability distribution
        rtol(float): relative tolerance
        breadth_first(bool): Flag to use breadth first search
        leaf_size(int): Uses a tree model to fit probability density.
        metric_params(dictionary): Used by tree model

        """

    def __init__(self, bandwidth=1.0, algorithm='auto', kernel='gaussian',
                 metric='euclidean', atol=0, rtol=0, breadth_first=True,
                 leaf_size=40, metric_params=None, num_cls=2):
        self.list_den_est = []
        self.num_cls = num_cls
        for i in range(num_cls):
            self.list_den_est.append(KernelDensity(bandwidth=bandwidth,
                                                   algorithm=algorithm,
                                                   kernel=kernel,
                                                   metric=metric,
                                                   atol=atol,
                                                   rtol=rtol,
                                                   breadth_first=breadth_first,
                                                   leaf_size=leaf_size,
                                                   metric_params=metric_params))

    def fit(self, x, y):
        """ Fits the kernel density estimates base on labelled data.
            Attr:
                x(ndarray[float]): N x 1 data array
                y(ndarray[float]): N x 1 label array
                Where N and c denotes number of samples and classes
                respectively. """

        classes = np.unique(y)
        # TODO: if train set lacks one of the classes ignore it

        cls_dep_x = [x[np.where(y == classes[i])[0]] for i in
                     range(self.num_cls)]

        for i in range(self.num_cls):
            # Reshape is required, otherwise there's ambiguity if it's one
            # sample with N dims or N samples with 1 dims

            dat = np.squeeze(cls_dep_x[i])
            dat = np.expand_dims(dat, axis=1)
            print(dat.shape)
            self.list_den_est[i].fit(dat)

    def transform(self, x):
        """ Calculates likelihop ods of given data.
            Attr:
                x(ndarray[float]): N x 1 data array
                Where N and c denotes number of samples and classes
            Return:
                 val(ndarray[float]): N x c  log-likelihood array
             respectively. """

        # Calculate likelihoods for each density estimate
        val = []
        for i in range(self.num_cls):
            dat = np.squeeze(x)
            dat = np.expand_dims(dat, axis=1)
            val.append(self.list_den_est[i].score_samples(dat))

        val = np.transpose(np.array(val))
        return val

    def fit_transform(self, x, y):
        self.fit(x, y)
        val = self.transform(x)
        return val
