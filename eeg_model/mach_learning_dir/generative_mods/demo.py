""" One dimensional KDE demo """

import numpy as np
from scipy.stats import norm, iqr

from function_density_estimation import KernelDensityEstimate


def test_kde(x, x_lsp, kernel='gaussian'):
    """ Select bandwidth of the gaussian kernel assuming data is also
    comming from a gaussian distribution.
    Ref: Silverman, Bernard W. Density estimation for statistics and data
    analysis. Vol. 26. CRC press, 1986. """
    bandwidth = 1.06 * min(
        np.std(x), iqr(x) / 1.34) * np.power(x.shape[0], -0.2)
    kde = KernelDensityEstimate(bandwidth=bandwidth, kernel=kernel).fit(x)
    log_density = kde.score_samples(x_lsp)
    return log_density

# N = 100
# np.random.seed(1)
# X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
#                     np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
#
# X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
#
# true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
#              + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))
#
# fig, ax = plt.subplots()
# ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
#         label='input distribution')
#
# for kernel in ['gaussian', 'tophat', 'epanechnikov']:
#     kde = KernelDensityEstimate(kernel=kernel, bandwidth=0.5).fit(X)
#     log_dens = kde.score_samples(X_plot)
#     ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
#             label="kernel = '{0}'".format(kernel))
#
# ax.text(6, 0.38, "N={0} points".format(N))
#
# ax.legend(loc='upper left')
# ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
#
# ax.set_xlim(-4, 9)
# ax.set_ylim(-0.02, 0.4)
# plt.show()
