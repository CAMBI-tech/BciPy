from sklearn.decomposition import PCA
from eeg_model.mach_learning.m_estimator.m_estimator import *


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
                x(ndarray[float]): C x N x k data array
                y(ndarray[int]): N x k observation (class) array
                    N is number of samples k is dimensionality of features
                    C is number of channels
                var_tol(float): Threshold to remove lower variance dims.
                """

        if var_tol:
            self.var_tol = var_tol

        for i in range(self.num_ch):
            self.list_pca[i].fit(x[i, :, :], y)
            max_sv = self.list_pca[i].singular_values_[0]
            self.list_pca[i].n_components = np.sum(self.list_pca[i].singular_values_ >= max_sv * self.var_tol)
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
                x(ndarray[float]): C x N x k data array
                y(ndarray[int]): N x k observation (class) array
                    N is number of samples k is dimensionality of features
                    C is number of channels
                var_tol(float): Threshold to remove lower variance dims.
            Return:
                y(ndarray(float)): N x ( sum_i (C x k')) data array
                    where k' is the new dimension for each PCA
                """

        self.fit(x, var_tol=var_tol)
        return self.transform(x)


class MPCA:
    # Channel wise MPCA

    def __init__(self, var_tol=1):
        self.var_tol = var_tol

    def fit(self, x, y=None, var_tol=None):
        """ Find channels wise robust covariances and apply pca.
            Args:
                x(ndarray[float]): C x N x k data array
                y(ndarray[int]): N x 1 observation (class) array
                    N is number of samples k is dimensionality of features
                    C is number of channels
                var_tol(float): Threshold to remove lower variance dims.
                """
        self.transform_matrix_list=[]
        q = .5
        p = x.shape[2]
        N = x.shape[1]

        c_square = sc.stats.chi2.ppf(q, p)
        b = sc.stats.chi2.cdf(c_square, p + 2) + c_square / p * (1 - sc.stats.chi2.cdf(c_square, p))

        for ch_index in range(x.shape[0]):
            X = x[ch_index]
            sample_mean = np.mean(X, axis=0)
            sample_sigma = 1. / N * np.dot(np.transpose(X - sample_mean), X - sample_mean)

            iteration = 0
            M_est_mean_new = sample_mean
            M_est_sigma_new = sample_sigma
            s_a_c = 1 # summed absolute change, initially large value
            while iteration < 1000 and s_a_c > .1**3:
                # print '{}/{}'.format(iteration, 1000)
                M_est_mean_old = M_est_mean_new
                M_est_sigma_old = M_est_sigma_new
                # update mean
                M_est_mean_new = mean_update(X=X, mean=M_est_mean_old, sigma_inv=np.linalg.inv(M_est_sigma_old), b=b, c_square=c_square)

                # update sigma
                M_est_sigma_new = sigma_update(X=X, mean=M_est_mean_new, sigma_inv=np.linalg.inv(M_est_sigma_old), b=b, c_square=c_square)

                s_a_c = np.sum(np.abs(M_est_mean_new-M_est_mean_old)) +\
                          np.sum(np.sum(np.abs(M_est_sigma_new-M_est_sigma_old)))
                # print s_a_c
                iteration += 1
                if iteration == 999:
                    print 'Max number of iterations reached for m estimation for pca. Last s_a_c: {}. If last s_a_c is large (>.1) use regular calibration.'.format(s_a_c)


            vals, vecs = eigsorted(M_est_sigma_new)

            lim = vals[0]*self.var_tol

            transform_matrix = []
            for index in range(len(vals)):
                if vals[index]>lim:
                    transform_matrix.append(vecs[:, index])

            self.transform_matrix_list.append(np.transpose(np.array(transform_matrix)))

    def transform(self, x, y=None):

        C, N, p = x.shape

        new_p = len(self.transform_matrix_list[0][0])
        new_x = np.zeros((C, N, new_p))
        for channel in range(len(x.shape[0])):
            new_x[channel, :, :] = np.dot(x[channel, :, :], self.transform_matrix_list[channel])



        return new_x

    def fit_transform(self, x, y=None, var_tol=None):
        """ Fits parameters wrt. the input matrix and outputs corresponding
            reduced form feature vector.
            Args:
                x(ndarray[float]): C x N x k data array
                y(ndarray[int]): N x k observation (class) array
                    N is number of samples k is dimensionality of features
                    C is number of channels
                var_tol(float): Threshold to remove lower variance dims.
            Return:
                y(ndarray(float)): N x ( sum_i (C x k')) data array
                    where k' is the new dimension for each PCA
                """

        self.fit(x, var_tol=var_tol)
        return self.transform(x)