import numpy as np
from sklearn.decomposition import PCA


class PrincipalComponentAnalysis(PCA):
    """ Principal component analysis implementation using the scikit-learn
        PCA object. The object is consistent, therefore kept as is.
        For reference check scikit-learn/decomposition/PCA """

    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None):
        super(PrincipalComponentAnalysis,
              self).__init__(n_components=n_components, copy=copy,
                             whiten=whiten,
                             svd_solver=svd_solver, tol=tol,
                             iterated_power=iterated_power,
                             random_state=random_state)
        self.var_tol = None

    def fit(self, x, y=None, var_tol=None):
        """ Inherits PCA fit() function from scikit-learn
            Args:
                x(ndarray[float]): N x k data array
                y(ndarray[int]): N x k observation (class) array
                    N is number of samples k is dimensionality of features
                var_tol(float): Threshold to remove lower variance dims.
                """

        if var_tol:
            self.var_tol = var_tol

        super(PrincipalComponentAnalysis, self).fit(x, y)
        if var_tol:
            max_var = self.explained_variance_ratio_[0]
            self.n_components = np.sum(
                np.array(
                    self.explained_variance_ratio_ >= max_var * self.var_tol))

        super(PrincipalComponentAnalysis, self).fit(x, y)

    def fit_transform(self, x, y=None, var_tol=None):
        """ Inherits PCA fit_transform() function from scikit-learn
            Args:
                x(ndarray[float]): N x k data array
                y(ndarray[int]): N x k observation (class) array
                    N is number of samples k is dimensionality of features
                var_tol(float): Threshold to remove lower variance dims.
                """
        if var_tol:
            self.var_tol = var_tol

        super(PrincipalComponentAnalysis, self).fit(x, y)
        if var_tol:
            max_var = self.explained_variance_ratio_[0]
            self.n_components = np.sum(
                np.array(
                    self.explained_variance_ratio_ >= max_var * self.var_tol))

        return super(PrincipalComponentAnalysis, self).fit_transform(x, y)

    def opt_param(self):
        return 0
