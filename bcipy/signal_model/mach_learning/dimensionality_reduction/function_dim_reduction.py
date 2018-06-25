from sklearn.decomposition import PCA
from bcipy.signal_model.mach_learning.m_estimator.m_estimator import eigsorted, robust_mean_covariance
import numpy as np


class ChannelWisePrincipalComponentAnalysis:
    """ Channel wise PCA application. Creates PCA objects respective to the
        number of channels.
        Attr:
            n_components(int): Number of components in PCA
            copy(bool): Saves the matrix if  True and updates on each fit()
            whiten(bool): Whitens the PCA matrix to form a tight frame
            svd_solver(string): SV Decomposition solver method
            tol=var_tol(float): Unfortunately I re-implemented it
                Tolerance to the singular values of the matrix
            random_state(seed): Random state seed
            num_ch(int): number of channels in expected data
            var_tol(float): Tolerance to the variance
         """

    def __init__(self, n_components=None, var_tol=0, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None, num_ch=1):

        self.list_pca = []
        for idx in range(num_ch):
            self.list_pca.append(PCA(n_components=n_components, copy=copy,
                                     whiten=whiten,
                                     svd_solver=svd_solver, tol=tol,
                                     iterated_power=iterated_power,
                                     random_state=random_state))
        self.var_tol = var_tol
        self.num_ch = num_ch

    def fit(self, x, y=None, var_tol=None):
        """ Inherits PCA fit() function from scikit-learn. Fits the
            transformation matrix wrt. tolerance to each PCA.
            Args:
                x(ndarray[float]): num_channels x num_samples x k data array
                y(ndarray[int]): num_samples x k observation (class) array
                    num_samples is number of samples k is dimensionality of features
                    num_channels is number of channels
                var_tol(float): Threshold to remove lower variance dims.
                """

        if var_tol:
            self.var_tol = var_tol

        for i in range(self.num_ch):
            self.list_pca[i].fit(x[i, :, :], y)
            max_sv = self.list_pca[i].singular_values_[0]
            self.list_pca[i].n_components = np.sum(self.list_pca[i].singular_values_ >= max_sv * self.var_tol)
            # self.list_pca[i].n_components = 5
            try:
                self.list_pca[i].fit(x[i, :, :], y)
            except Exception as e:
                raise e

    def transform(self, x, y=None):
        f_vector = []
        for i in range(self.num_ch):
            # TODO: Observe that scikit learn PCA does not accept y
            f_vector.append(self.list_pca[i].transform(x[i, :, :]))

        return np.concatenate(f_vector, axis=1)

    def fit_transform(self, x, y=None, var_tol=None):
        """ Fits parameters wrt. the input matrix and outputs corresponding
            reduced form feature vector.
            Args:
                x(ndarray[float]): num_channels x num_samples x k data array
                y(ndarray[int]): num_samples x k observation (class) array
                    num_samples is number of samples k is dimensionality of features
                    num_channels is number of channels
                var_tol(float): Threshold to remove lower variance dims.
            Return:
                y(ndarray(float)): num_samples x ( sum_i (num_channels x k')) data array
                    where k' is the new dimension for each PCA
                """

        self.fit(x, var_tol=var_tol)
        return self.transform(x)


class MPCA:
    """ Channel wise MPCA
        attributes:
            var_tol(float): Variance tolerance with respect to principal component(eigen value)
    """

    def __init__(self, var_tol=1):
        self.var_tol = var_tol
        self.transform_matrix_list = []
        for z in range(10):  # 10 for 10 fold cross validation
            self.transform_matrix_list.append([])
        self.current_fold = -1  # if -1, uses complete data, else uses data when fold i is removed.

    def fit(self, data, y=None, var_tol=None):
        """ Find channel wise robust covariances and apply pca.
            Args:
                data(ndarray[float]): num_channels x num_samples x k data array
                y(ndarray[int]): num_samples x 1 observation (class) array
                    num_samples is number of samples k is dimensionality of features
                    num_channels is number of channels
                var_tol(float): Threshold to remove lower variance dims.
                """

        num_channels, num_samples, num_features = data.shape

        if not self.transform_matrix_list[self.current_fold]:  # if does not exist

            for channel in range(num_channels):
                X_channel = data[channel]

                M_est_mean, M_est_sigma = robust_mean_covariance(data=X_channel)
                vals, vecs = eigsorted(M_est_sigma)

                lim = vals[0]*self.var_tol

                transform_matrix = []
                # for index in range(len(vals)):
                for index in range(5):
                    if vals[index] > lim:
                        transform_matrix.append(vecs[:, index])

                self.transform_matrix_list[self.current_fold].append(np.transpose(np.array(transform_matrix)))

    def transform(self, data, y=None):

        num_channels, num_samples, num_features = data.shape

        new_data = []
        for channel in range(num_channels):
            new_data.append(np.dot(data[channel], self.transform_matrix_list[self.current_fold][channel]))

        concatenated_new_x = []

        for sample_index in range(num_samples):
            new_sample = []
            for channel in range(num_channels):
                new_sample = np.append(new_sample, new_data[channel][sample_index])
            concatenated_new_x.append(new_sample)

        return np.array(concatenated_new_x)

    def fit_transform(self, data, y=None, var_tol=None):
        """ Fits parameters wrt. the input matrix and outputs corresponding
            reduced form feature vector.
            Args:
                data(ndarray[float]): num_channels x num_samples x k data array
                y(ndarray[int]): num_samples x k observation (class) array
                    num_samples is number of samples k is dimensionality of features
                    num_channels is number of channels
                var_tol(float): Threshold to remove lower variance dims.
            Return:
                y(ndarray(float)): num_samples x ( sum_i (num_channels x k')) data array
                    where k' is the new dimension for each PCA
                """

        self.fit(data, var_tol=var_tol)
        return self.transform(data)
