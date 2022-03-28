import numpy as np
from sklearn.decomposition import PCA


class ChannelWisePrincipalComponentAnalysis:
    """Channel wise PCA application. Creates PCA objects respective to the
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

    def __init__(
        self,
        n_components=None,
        var_tol=0,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
        num_ch=1,
    ):

        self.list_pca = []
        for idx in range(num_ch):
            self.list_pca.append(
                PCA(
                    n_components=n_components,
                    copy=copy,
                    whiten=whiten,
                    svd_solver=svd_solver,
                    tol=tol,
                    iterated_power=iterated_power,
                    random_state=random_state,
                )
            )
        self.var_tol = var_tol
        self.num_ch = num_ch

    def fit(self, x, y=None, var_tol=None):
        """Inherits PCA fit() function from scikit-learn. Fits the
        transformation matrix wrt. tolerance to each PCA.
        Args:
            x(ndarray[float]): C x N x k data array
            y(ndarray[int]): N x k observation (class) array
                N is number of samples k is dimensionality of features
                C is number of channels
            var_tol(float): Threshold to remove lower variance dims.
        """

        for i in range(self.num_ch):
            self.list_pca[i].fit(x[i, :, :], y)

    def transform(self, x, y=None):
        f_vector = []
        for i in range(self.num_ch):
            f_vector.append(self.list_pca[i].transform(x[i, :, :]))

        return np.concatenate(f_vector, axis=1)

    def fit_transform(self, x, y=None, var_tol=None):
        """Fits parameters wrt. the input matrix and outputs corresponding
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
