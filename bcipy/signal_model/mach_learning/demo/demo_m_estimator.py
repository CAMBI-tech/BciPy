from bcipy.signal_model.mach_learning.m_estimator.m_estimator import mean_update, sigma_update, eigsorted
import numpy as np
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy as sc


def demo_m_estimator():
    # Simple demo for m estimators.

    N = 150
    p = 2
    q = .5
    true_mean = 0 + 0*np.random.standard_normal(p)
    true_sigma = make_spd_matrix(n_dim=p)

    # Outliers have a completely different distribution.
    N_outlier = int(N*.05)
    true_mean_outlier = 3 + 0*np.random.standard_normal(p)
    true_sigma_outlier = make_spd_matrix(n_dim=p)

    X_positive = np.random.multivariate_normal(mean=true_mean, cov=true_sigma, size=N-N_outlier)
    X_outlier = np.random.multivariate_normal(mean=true_mean_outlier, cov=true_sigma_outlier, size=N_outlier)
    X = np.vstack((X_positive, X_outlier))

    sample_mean = np.mean(X, axis=0)
    sample_sigma = 1./N*np.dot(np.transpose(X-sample_mean), X-sample_mean)

    print('Sample mean - true mean:\n', sample_mean - true_mean)
    print('Sample cov - true cov:\n', sample_sigma - true_sigma)

    # inverse CDF of chi2 with p degrees of freedom at q'th quantile
    c_square = sc.stats.chi2.ppf(q, p)
    b = sc.stats.chi2.cdf(c_square, p + 2) + c_square/p*(1 - sc.stats.chi2.cdf(c_square, p))

    iteration = 0
    M_est_mean_new = sample_mean
    M_est_sigma_new = sample_sigma
    s_a_c = 4 # summed absolute change, initially large value

    while iteration < 1000 and s_a_c > .1**6:
        # print '{}/{}'.format(iteration, 1000)
        M_est_mean_old = M_est_mean_new
        M_est_sigma_old = M_est_sigma_new
        # update mean
        M_est_mean_new = mean_update(X=X, mean=M_est_mean_old, sigma_inv=np.linalg.inv(M_est_sigma_old), b=b, c_square=c_square)

        # update sigma
        M_est_sigma_new = sigma_update(X=X, mean=M_est_mean_new, sigma_inv=np.linalg.inv(M_est_sigma_old), b=b, c_square=c_square)

        s_a_c = np.sum(np.abs(M_est_mean_new-M_est_mean_old)) +\
                  np.sum(np.sum(np.abs(M_est_sigma_new-M_est_sigma_old)))
        print(s_a_c)
        iteration += 1

    print('\n\nFinished in {} iterations.\n\n'.format(iteration))
    print('M estimate mean - true mean:\n', M_est_mean_new - true_mean)
    print('M estimate sigma - true sigma:\n', M_est_sigma_new - true_sigma)
    print('\nΣ|Sample mean - true mean|:', np.sum(np.abs(sample_mean - true_mean)))
    print('Σ|M estimate mean - true mean|:', np.sum(np.abs(M_est_mean_new - true_mean)))
    print('Σ|Sample cov - true cov|:', np.sum(np.sum(np.abs(sample_sigma - true_sigma))))
    print('Σ|M estimate sigma - true sigma|:', np.sum(np.sum(np.abs(M_est_sigma_new - true_sigma))))

    # If data is two dimensional plot the results.
    if p == 2:
        plt.scatter(X_positive[:, 0], X_positive[:, 1], 1)
        plt.hold(True)
        plt.scatter(X_outlier[:, 0], X_outlier[:, 1], c='red', s=1)

        nstd = 1

        # Sample estimate's ellipse
        ax = plt.subplot(111)
        cov = sample_sigma
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * nstd * np.sqrt(vals)
        ell = Ellipse(xy=(sample_mean[0], sample_mean[1]),
                      width=w, height=h,
                      angle=theta, color='red')
        ell.set_facecolor('none')
        ax.add_artist(ell)

        cov = M_est_sigma_new
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * nstd * np.sqrt(vals)
        ell = Ellipse(xy=(M_est_mean_new[0], M_est_mean_new[1]),
                      width=w, height=h,
                      angle=theta, color='blue')
        ell.set_facecolor('none')
        ax.add_artist(ell)

        cov = true_sigma
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * nstd * np.sqrt(vals)
        ell = Ellipse(xy=(true_mean[0], true_mean[1]),
                      width=w, height=h,
                      angle=theta, color='black')
        ell.set_facecolor('none')
        ax.add_artist(ell)

        plt.grid()
        plt.show()

