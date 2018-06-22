# -*- coding: utf-8 -*-
import numpy as np
import scipy as sc


def eigsorted(cov):
    """
    Find and sort Eigenvalues and vectors
    :param cov: covariance matrix
    :return: sorted Eigenvalues and vectors
    """

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def u_func(mahalanobis_dist, huber_denom, c_square):
    """
    Huber's loss function
    :param mahalanobis_dist: Mahalanobis distance
    :param huber_denom: A constant in p (number of features)
    :param c_square: A constant in p and q (trade-off variable)
    :return: weight of sample
    """

    if mahalanobis_dist <= c_square:
        return 1./huber_denom
    else:
        return c_square/(mahalanobis_dist*huber_denom)


def mean_update(data, mean, sigma_inv, huber_denom, c_square):
    """
    Update mean estimate
    :param data: Data
    :param mean: Current mean
    :param sigma_inv: Current covariance inverse
    :param huber_denom: A constant in p (number of features)
    :param c_square: A constant in p and q (trade-off variable)
    :return: New mean
    """

    N, p = data.shape
    mean_hat = np.zeros(p)
    sum_u = 0
    for z in range(N):

        mahalanobis_dist = np.dot(np.dot(data[z] - mean, sigma_inv), data[z] - mean)
        u = u_func(mahalanobis_dist=mahalanobis_dist, huber_denom=huber_denom, c_square=c_square)

        sum_u += u
        mean_hat += u*data[z]

    mean_hat = mean_hat/sum_u

    return mean_hat


def sigma_update(data, mean, sigma_inv, huber_denom, c_square):
    """
    Update sigma estimate
    :param data: Data
    :param mean: Current mean
    :param sigma_inv: Current covariance inverse
    :param huber_denom: A constant in p (number of features)
    :param c_square: A constant in p and q (trade-off variable)
    :return: New sigma
    """

    N, p = data.shape
    sigma_hat = np.zeros((p,p))

    for z in range(N):
        sigma_hat += 1./N*u_func(mahalanobis_dist=np.dot(np.dot(data[z] - mean, sigma_inv), data[z] - mean), huber_denom=huber_denom, c_square=c_square)*np.outer(data[z]- mean, data[z]- mean)

    return sigma_hat


def robust_mean_covariance(data, q=.85):
    """
    Use m estimation theory to find robust mean and covariance for data data.
    http://www.spg.tu-darmstadt.de/media/spg/ieee_ssrsp/material/SummerSchool_Ollila.pdf
    or checkout paper by Berkan Kadioglu 'M estimation based subspace learning for brain computer interfaces'

    :param data: Data matrix with dimensions Nxp
    :param q: Trade-off variable between regular covariance and weighted covariance.
    :return: Tuple of mean and covariance for data
    """

    N, p = data.shape  # N: number of samples, p: number of features
    c_square = sc.stats.chi2.ppf(q, p)
    huber_denom = sc.stats.chi2.cdf(c_square, p + 2) + c_square / p * (1 - sc.stats.chi2.cdf(c_square, p))

    sample_mean = np.mean(data, axis=0)
    sample_sigma = 1. / N * np.dot(np.transpose(data - sample_mean), data - sample_mean)

    M_est_mean_new = sample_mean
    M_est_sigma_new = sample_sigma

    iteration = 0

    s_a_c = 2  # summed absolute change, initially large value
    while iteration < 1000 and s_a_c > 1:

        M_est_mean_old = M_est_mean_new
        M_est_sigma_old = M_est_sigma_new
        # update mean
        M_est_mean_new = mean_update(data=data, mean=M_est_mean_old, sigma_inv=np.linalg.inv(M_est_sigma_old), huber_denom=huber_denom,
                                     c_square=c_square)

        # update sigma
        M_est_sigma_new = sigma_update(data=data, mean=M_est_mean_new, sigma_inv=np.linalg.inv(M_est_sigma_old), huber_denom=huber_denom,
                                       c_square=c_square)

        s_a_c = np.sum(np.abs(M_est_mean_new - M_est_mean_old)) + \
                np.sum(np.sum(np.abs(M_est_sigma_new - M_est_sigma_old)))
        # print s_a_c
        iteration += 1
        if iteration == 999 and s_a_c > 1:
            print('Max number of iterations reached for m estimation. Last s_a_c: {}.'.format(s_a_c))
            print('It is advised to have at least 120 positive trials in robust calibration.')

    return M_est_mean_new, M_est_sigma_new
