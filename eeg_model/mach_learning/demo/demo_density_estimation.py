""" One dimensional KDE demo """

import numpy as np
from scipy.stats import norm, iqr


def _demo_kde():
    from eeg_model.mach_learning.generative_mods.function_density_estimation \
        import KernelDensityEstimate
    import matplotlib.pyplot as plt

    n = 100
    np.random.seed(1)
    x = np.concatenate((np.random.normal(0, 1, int(0.3 * n)),
                        np.random.normal(5, 1, int(0.7 * n))))[:, np.newaxis]

    y = np.zeros(x.shape)

    x_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

    true_dens = (0.3 * norm(0, 1).pdf(x_plot[:, 0])
                 + 0.7 * norm(5, 1).pdf(x_plot[:, 0]))

    fig, ax = plt.subplots()
    ax.fill(x_plot[:, 0], true_dens, fc='black', alpha=0.2,
            label='input distribution')

    bandwidth = 1.06 * min(
        np.std(x), iqr(x) / 1.34) * np.power(x.shape[0], -0.2)

    for kernel in ['gaussian', 'tophat', 'epanechnikov']:
        kde = KernelDensityEstimate(kernel=kernel, bandwidth=bandwidth,
                                    num_cls=1)
        kde.fit(x, y)
        log_dens = kde.list_den_est[0].score_samples(x_plot)
        ax.plot(x_plot[:, 0], np.exp(log_dens), '-',
                label="kernel = '{0}'".format(kernel))

    ax.text(6, 0.38, "N={0} points".format(n))

    ax.legend(loc='upper left')
    ax.plot(x[:, 0], -0.005 - 0.01 * np.random.random(x.shape[0]), '+k')

    ax.set_xlim(-4, 9)
    ax.set_ylim(-0.02, 0.4)
    plt.show()
    print('KDE Flows!')
    return 0


def main():
    _demo_kde()

    return 0


if __name__ == "__main__":
    main()
