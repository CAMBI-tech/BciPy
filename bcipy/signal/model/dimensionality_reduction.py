import logging

from typing import Optional
import numpy as np
from sklearn.decomposition import PCA

from bcipy.config import SESSION_LOG_FILENAME


class ChannelWisePrincipalComponentAnalysis:
    """Creates a PCA object for each channel.

    Args:
        n_components(int or float): Number of components in PCA
            - If 0 < n_components < 1, keeps enough components to preserve the specified fraction of variance.
            - If int, keeps the first n_components components.
            - If None, keeps all components

            NOTE - Keeping more components (or using a float closer to 1) preserves more structure;
            keeping fewer (or using a float closer to 0) achieves more compression and thus more speedup for
            downstream steps. The tradeoff between compression and performance is fundamentally arbitrary.
            To choose number of components, try plotting the explained_variance_ratio_ attribute of the PCA object
            on a representative dataset, and check the effect on runtime and task performance.

        random_state(int): Random state seed
        num_ch(int): number of channels in expected data
    """

    def __init__(self, n_components: Optional[float] = None, random_state: Optional[int] = None, num_ch: int = 1):
        self.num_ch = num_ch
        self.list_pca = [PCA(n_components=n_components, random_state=random_state) for _ in range(self.num_ch)]
        self.logger = logging.getLogger(SESSION_LOG_FILENAME)
        self.logger.info(f"PCA. n_components={n_components}, random_state={random_state}, num_ch={num_ch}")

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit PCA to each channel of data.

        Args:
            x(ndarray[float]): C x N x k data array
            y(ndarray[int]): N x k observation (class) array
                N is number of samples k is dimensionality of features
                C is number of channels
        """
        for i in range(self.num_ch):
            self.list_pca[i].fit(x[i, :, :], y)

    def transform(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply fitted PCA to each channel.

        Args:
            x: data, with shape (channels, items, samples)
            y: labels (ignored)

        Returns:
            reduced dim data, shape (items, K), where K is the size after reducing and concatenating all channels
        """
        f_vector = []
        for i in range(self.num_ch):
            f_vector.append(self.list_pca[i].transform(x[i, :, :]))
        return np.concatenate(f_vector, axis=1)

    def fit_transform(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class MockPCA:
    """Mock PCA object for testing purposes."""

    def __init__(self, *args, **kw):
        pass

    def fit(self, *args, **kw):
        pass

    def transform(self, x: np.ndarray, y: Optional[np.ndarray] = None, var_tol: Optional[float] = None) -> np.ndarray:
        """Mock transform method for testing purposes.

        Args:
            x: data, with shape (channels, items, samples)
            y: labels (ignored)
            var_tol: variance tolerance (ignored)

        Returns:
            reduced dim data, shape (items, K), where K is the size after reducing and concatenating all channels
        """
        x = x.transpose([1, 0, 2])
        x = x.reshape([x.shape[0], -1])
        return x

    def fit_transform(self, x: np.ndarray, y: Optional[np.ndarray]
                      = None, var_tol: Optional[float] = None) -> np.ndarray:
        self.fit(x)
        return self.transform(x, y, var_tol)
