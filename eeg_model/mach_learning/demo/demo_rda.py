import numpy as np


def _demo_rda():
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
    _demo_rda()

    return 0


if __name__ == "__main__":
    main()
