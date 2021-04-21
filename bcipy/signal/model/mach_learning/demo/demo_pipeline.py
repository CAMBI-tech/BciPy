import matplotlib.pylab as plt
from bcipy.signal.model.mach_learning.density_estimation import KernelDensityEstimate
from bcipy.signal.model.mach_learning.classifier import RegularizedDiscriminantAnalysis
from bcipy.signal.model.mach_learning.dimensionality_reduction import ChannelWisePrincipalComponentAnalysis
from bcipy.signal.model.mach_learning.pipeline import Pipeline
import numpy as np
from scipy.stats import iqr
import time
import matplotlib as mpl

mpl.use('TkAgg')


def _demo_pipeline():
    dim_x = 2
    num_x_p = 200
    num_x_n = 200
    var_tol = 0.8
    num_ch = 2

    mtx_p = [np.array([[1, 0], [0, 1]]), np.array([[1, 2], [2, 1]])]
    mtx_n = [np.array([[2, 0], [0, 2]]), np.array([[1, -2], [-2, 1]])]

    x_p = np.asarray([np.dot(np.random.randn(num_x_p, dim_x), mtx_p[i]) for i in range(num_ch)])
    x_n = 3 + np.array([np.dot(np.random.randn(num_x_p, dim_x), mtx_n[i]) for i in range(num_ch)])
    y_p = [1] * num_x_p
    y_n = [0] * num_x_n

    x = np.concatenate((x_n, x_p), axis=1)
    y = np.concatenate((y_n, y_p), axis=0)

    permutation = np.random.permutation(x.shape[1])
    x = x[:, permutation, :]
    y = y[permutation]

    """ Select bandwidth of the gaussian kernel assuming data is also
        comming from a gaussian distribution.
        Ref: Silverman, Bernard W. Density estimation for statistics and data
        analysis. Vol. 26. CRC press, 1986. """
    bandwidth = 1.06 * min(np.std(x), iqr(x) / 1.34) * np.power(x.shape[0], -0.2)

    pca = ChannelWisePrincipalComponentAnalysis(num_ch=x.shape[0])
    rda = RegularizedDiscriminantAnalysis()
    kde = KernelDensityEstimate(bandwidth=bandwidth)

    model = Pipeline()
    model.add(pca)
    model.add(rda)
    model.add(kde)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(212)
    ax_2 = fig.add_subplot(221)
    ax_3 = fig.add_subplot(222)

    for gam in [0, 0.3, 0.6, 0.9]:
        for lam in [0, 0.3, 0.6, 0.9]:
            model.pipeline[1].lam = lam
            model.pipeline[1].gam = gam

            if gam == 0 and lam == 0:
                # Show this once only bad implementation but I don't care
                model.pipeline[0].var_tol = 0
                model.fit(x, y)
                sv_init = [
                    model.pipeline[0].list_pca[i].singular_values_ for i in range(len(model.pipeline[0].list_pca))
                ]
                model.pipeline[0].var_tol = var_tol
                model.fit(x, y)
                sv_final = [
                    model.pipeline[0].list_pca[i].singular_values_ for i in range(len(model.pipeline[0].list_pca))
                ]
                print(f'Initial SV:{sv_init}')
                print(f'-- using tolerance:{var_tol} -->')
                print(f'Final SV:{sv_final}')

                print(f'Init dim.:{x.shape} -> Final dim.:{model.line_el[1].shape}')

            model.fit_transform(x, y)

            el = model.line_el[1]
            x_min, x_max = el[:, 0].min() - 1, el[:, 0].max() + 1
            y_min, y_max = el[:, 1].min() - 1, el[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
            z = model.pipeline[1].predict(np.c_[xx.ravel(), yy.ravel()])
            z = z.reshape(xx.shape)

            ax.clear()
            ax_2.clear()
            ax_3.clear()
            ax.contourf(xx, yy, z, alpha=0.2, c=y, s=20)

            ax.scatter(model.line_el[1][y == 1, 0], model.line_el[1][y == 1, 1], c='r')
            ax.scatter(model.line_el[1][y == 0, 0], model.line_el[1][y == 0, 1], c='g')
            ax.set_title('after PCA')

            ax_2.scatter(x[0, y == 1, 0], x[0, y == 1, 1], c='r')
            ax_2.scatter(x[0, y == 0, 0], x[0, y == 0, 1], c='g')

            ax_3.scatter(x[1, y == 1, 0], x[1, y == 1, 1], c='r')
            ax_3.scatter(x[1, y == 0, 0], x[1, y == 0, 1], c='g')
            ax_2.set_title('1st dim')
            ax_3.set_title('2nd dim')

            fig.canvas.draw()

            time.sleep(0.2)

    time.sleep(1)
    plt.ioff()
    fig_2, axn = plt.subplots()
    x_plot = np.linspace(np.min(model.line_el[-1]), np.max(model.line_el[-1]), 1000)[:, np.newaxis]
    axn.plot(
        model.line_el[2][y == 0],
        -0.005 - 0.01 * np.random.random(model.line_el[2][y == 0].shape[0]),
        'ro',
        label='class(-)',
    )
    axn.plot(
        model.line_el[2][y == 1],
        -0.005 - 0.01 * np.random.random(model.line_el[2][y == 1].shape[0]),
        'go',
        label='class(+)',
    )
    for idx in range(len(model.pipeline[2].list_den_est)):
        log_dens = model.pipeline[2].list_den_est[idx].score_samples(x_plot)
        axn.plot(x_plot[:, 0], np.exp(log_dens), 'r-' * (idx == 0) + 'g--' * (idx == 1), linewidth=2.0)
    axn.legend(loc='upper right')
    plt.title('Likelihoods Given the Labels')
    plt.ylabel('p(e|l)')
    plt.xlabel('scores')
    fig_2.show()
    time.sleep(10)


if __name__ == "__main__":
    _demo_pipeline()
