""" One dimensional KDE demo """

import numpy as np
from scipy.stats import norm, iqr


def test_kde(x, x_lsp, kernel='gaussian'):
    """ Select bandwidth of the gaussian kernel assuming data is also
    comming from a gaussian distribution.
    Ref: Silverman, Bernard W. Density estimation for statistics and data
    analysis. Vol. 26. CRC press, 1986. """
    from function_density_estimation import KernelDensityEstimate
    bandwidth = 1.06 * min(
        np.std(x), iqr(x) / 1.34) * np.power(x.shape[0], -0.2)

    # For demo purposes assume there's only 1 class
    y = np.zeros(x.shape)

    kde = KernelDensityEstimate(bandwidth=bandwidth, kernel=kernel, num_cls=1)
    kde.fit(x, y)
    log_density = kde.transform(x_lsp)
    return log_density


def _test_kde():
    from eeg_model.mach_learning.generative_mods.function_density_estimation \
        import KernelDensityEstimate
    import matplotlib.pyplot as plt

    N = 100
    np.random.seed(1)
    X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                        np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]

    Y = np.zeros(X.shape)

    X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

    true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
                 + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

    fig, ax = plt.subplots()
    ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
            label='input distribution')

    for kernel in ['gaussian', 'tophat', 'epanechnikov']:
        kde = KernelDensityEstimate(kernel=kernel, bandwidth=0.5, num_cls=1)
        kde.fit(X, Y)
        log_dens = kde.list_den_est[0].score_samples(X_plot)
        ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
                label="kernel = '{0}'".format(kernel))

    ax.text(6, 0.38, "N={0} points".format(N))

    ax.legend(loc='upper left')
    ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

    ax.set_xlim(-4, 9)
    ax.set_ylim(-0.02, 0.4)
    plt.show()
    print('KDE Flows!')
    return 0


def main():
    _test_kde()

    return 0


if __name__ == "__main__":
    main()
