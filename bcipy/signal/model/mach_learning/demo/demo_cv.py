import numpy as np
from bcipy.signal.model.mach_learning.cross_validation import cross_validation
from bcipy.signal.model.mach_learning.classifier import RegularizedDiscriminantAnalysis
from bcipy.signal.model.mach_learning.dimensionality_reduction import ChannelWisePrincipalComponentAnalysis
from bcipy.signal.model.mach_learning.pipeline import Pipeline


def _demo_cv():
    dim_x = 2
    num_ch = 2
    num_x_p = 2000
    num_x_n = 500

    x_p = 2 * np.random.randn(num_ch, num_x_p, dim_x)
    x_n = 4 + np.random.randn(num_ch, num_x_n, dim_x)
    y_p = [1] * num_x_p
    y_n = [0] * num_x_n

    x = np.concatenate((x_p, x_n), 1)
    y = np.concatenate(np.asarray([y_p, y_n]), 0)
    permutation = np.random.permutation(x.shape[1])
    x = x[:, permutation, :]
    y = y[permutation]

    rda = RegularizedDiscriminantAnalysis()
    pca = ChannelWisePrincipalComponentAnalysis(num_ch=num_ch)
    pipeline = Pipeline()
    pipeline.add(pca)
    pipeline.add(rda)
    _ = cross_validation(x, y, pipeline)
    print('Cross Validation Flows!')


if __name__ == "__main__":
    _demo_cv()
