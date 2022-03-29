import numpy as np
from sklearn.decomposition import PCA


class ChannelWisePrincipalComponentAnalysis:
    def __init__(
        self,
        n_components,
        num_ch,
        random_state=None,
    ):
        """Channel wise PCA application. Creates PCA objects respective to the
        number of channels.
        Attr:
            n_components(int or float): Number of components in PCA.
            if int, indicates number of components to keep.
            if float, indicates percentage of total variance to keep.
            num_ch(int): number of channels in expected data
            random_state(seed): Random state seed
        """
        self.num_ch = num_ch
        self.list_pca = [PCA(n_components=n_components, random_state=random_state) for _ in range(self.num_ch)]

    def fit(self, x, y=None):
        """Fits PCA to each channel of data x.
        Args:
            x(ndarray[float]): C x N x k data array
            y(ndarray[int]): N x k observation (class) array
                N is number of samples
                k is number of features
                C is number of channels
        """
        for i in range(self.num_ch):
            self.list_pca[i].fit(x[i, :, :], y)

    def transform(self, x, y=None):
        f_vector = []
        for i in range(self.num_ch):
            f_vector.append(self.list_pca[i].transform(x[i, :, :]))

        return np.concatenate(f_vector, axis=1)

    def fit_transform(self, x, y=None):
        """See self.fit()

        Return:
            y(ndarray(float)): N x ( sum_i (C x k')) data array
                where k' is the new dimension for each PCA
        """
        self.fit(x)
        return self.transform(x)
