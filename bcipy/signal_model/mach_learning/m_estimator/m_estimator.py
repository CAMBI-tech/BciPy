# -*- coding: utf-8 -*-
import numpy as np
import scipy as sc


def eigsorted(cov):
    """
    Find and sort eigen values and vectors
    :param cov: covariance matrix
    :return: sorted eigen values and vectors
    """

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def u_func(t, b, c_square):
    """
    Huber's loss function
    :param t: Mahalanobis distance
    :param b: a constant in p
    :param c_square: a constant in p and q
    :return: weight of sample
    """

    if t <= c_square:
        return 1./b
    else:
        return c_square/(t*b)


def mean_update(X, mean, sigma_inv, b, c_square):
    """
    Update mean estimate
    :param X: Data
    :param mean: current mean
    :param sigma_inv: current covariance inverse
    :param b: a constant in p
    :param c_square: a constant in p and q
    :return: new mean
    """

    N, p = X.shape
    mean_hat = np.zeros(p)
    sum_u = 0
    for z in range(N):

        t = np.dot(np.dot(X[z] - mean, sigma_inv), X[z] - mean)
        u = u_func(t=t, b=b, c_square=c_square)

        sum_u += u
        mean_hat += u*X[z]

    mean_hat = mean_hat/sum_u

    return mean_hat


def sigma_update(X, mean, sigma_inv, b, c_square):
    """
    Update sigma estimate
    :param X: Data
    :param mean: current mean
    :param sigma_inv: current covariance inverse
    :param b: a constant in p
    :param c_square: a constant in p and q
    :return: new sigma
    """

    N, p = X.shape
    sigma_hat = np.zeros((p,p))

    for z in range(N):
        sigma_hat += 1./N*u_func(t=np.dot(np.dot(X[z] - mean, sigma_inv), X[z] - mean), b=b, c_square=c_square)*np.outer(X[z]- mean, X[z]- mean)

    return sigma_hat


def robust_mean_covariance(X, q=.85):
    """
    Use m estimation theory to find robust mean and covariance for data X.
    http://www.spg.tu-darmstadt.de/media/spg/ieee_ssrsp/material/SummerSchool_Ollila.pdf
    or checkout paper by Berkan Kadioglu 'M estimation based subspace learning for brain computer interfaces'

    :param X: Data matrix with dimensions Nxp
    :param q: Trade-off variable between regular covariance and weighted covariance.
    :return: Tuple of mean and covariance for X
    """

    N, p = X.shape  # N: number of samples, p: number of features
    c_square = sc.stats.chi2.ppf(q, p)
    b = sc.stats.chi2.cdf(c_square, p + 2) + c_square / p * (1 - sc.stats.chi2.cdf(c_square, p))

    sample_mean = np.mean(X, axis=0)
    sample_sigma = 1. / N * np.dot(np.transpose(X - sample_mean), X - sample_mean)

    M_est_mean_new = sample_mean
    M_est_sigma_new = sample_sigma


    iteration = 0
    s_a_c = 2  # summed absolute change, initially large value
    while iteration < 1000 and s_a_c > 1:
        # print '{}/{}'.format(iteration, 1000)
        M_est_mean_old = M_est_mean_new
        M_est_sigma_old = M_est_sigma_new
        # update mean
        M_est_mean_new = mean_update(X=X, mean=M_est_mean_old, sigma_inv=np.linalg.inv(M_est_sigma_old), b=b,
                                     c_square=c_square)

        # update sigma
        M_est_sigma_new = sigma_update(X=X, mean=M_est_mean_new, sigma_inv=np.linalg.inv(M_est_sigma_old), b=b,
                                       c_square=c_square)

        s_a_c = np.sum(np.abs(M_est_mean_new - M_est_mean_old)) + \
                np.sum(np.sum(np.abs(M_est_sigma_new - M_est_sigma_old)))
        # print s_a_c
        iteration += 1
        if iteration == 999 and s_a_c > 1:
            print 'Max number of iterations reached for m estimation. Last s_a_c: {}.'.format(s_a_c)
            print 'It is advised to have at least 120 positive trials in robust calibration.'

    return M_est_mean_new, M_est_sigma_new
