import numpy as np
from signal_model.mach_learning.cross_validation import cross_validation
from signal_model.mach_learning.classifier.function_classifier \
    import RegularizedDiscriminantAnalysis
from signal_model.mach_learning.dimensionality_reduction.function_dim_reduction \
    import DummyDimReduction, ChannelWisePrincipalComponentAnalysis
from signal_model.mach_learning.pipeline import Pipeline


def test_cv(x, y):
    rda = RegularizedDiscriminantAnalysis()
    pca = DummyDimReduction()
    pipeline = Pipeline()
    pipeline.add(pca)
    pipeline.add(rda)
    arg = cross_validation(x, y, pipeline)

    return arg


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

    arg_0 = test_cv(x, y)

    rda = RegularizedDiscriminantAnalysis()
    pca = DummyDimReduction()
    pipeline = Pipeline()
    pipeline.add(pca)
    pipeline.add(rda)
    arg_1 = cross_validation(x, y, pipeline)
    print('Cross Validation Flows!')

    return 0

if __name__ == "__main__":
    _demo_cv()
