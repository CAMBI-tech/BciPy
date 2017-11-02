import numpy as np


def test_rda(x, y, z):
    """ Test function called by MATLAB.
        Please refer to rda_consistency_test
        Args:
            x(ndarray[float]): N x k data array
            y(ndarray[int]): N x 1 observation (class) array
            z(ndarray[float]): M x k validation array
                N, M is number of samples k is dimensionality of features
        Return:
            rda.cov(ndarray[float]): covariances
            rda.reg_inverse_cov(ndarray[float]): regularized Covariances
            rda.means(ndarray[float]): means
            prb(ndarray[float]): priors
                for each class
            """
    from function_classifier import RegularizedDiscriminantAnalysis
    rda = RegularizedDiscriminantAnalysis()
    rda.fit(x, y)

    prb = rda.get_proba(z)

    return [np.array(rda.cov), np.array(rda.reg_inverse_cov), np.array(
        rda.means), prb]


def _test_rda():
    from eeg_model.mach_learning.classifier.function_classifier import \
        RegularizedDiscriminantAnalysis
    dim_x = 2
    num_x_p = 2000
    num_x_n = 500

    x_p = 2 * np.random.randn(num_x_p, dim_x)
    x_n = 4 + np.random.randn(num_x_n, dim_x)
    y_p = [1] * num_x_p
    y_n = [0] * num_x_n

    x = np.concatenate(np.asarray([x_p, x_n]), 0)
    y = np.concatenate(np.asarray([y_p, y_n]), 0)

    # out = test_rda(x, y)

    rda = RegularizedDiscriminantAnalysis()

    z = rda.fit_transform(x, y)
    print('Successfully used RDA')
    rda.fit(x, y)
    z_2 = rda.transform(x)
    print('MSE:{} for fit_transform'.format(np.sum(np.abs(z_2 - z))))

    return 0


def main():
    _test_rda()

    return 0


if __name__ == "__main__":
    main()
