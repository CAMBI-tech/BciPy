# -*- coding: utf-8 -*-
import numpy as np
import scipy as sc
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import warnings
import matlab.engine

from helpers.load import read_data_csv, load_experimental_data
from acquisition.sig_pro.sig_pro import sig_pro
from eeg_model.mach_learning.train_model import train_pca_rda_kde_model
from eeg_model.mach_learning.trial_reshaper import trial_reshaper
from helpers.data_viz import generate_offline_analysis_screen
from helpers.triggers import trigger_decoder
import pickle


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


def u_func(t, b, c_square):

    if t <= c_square:
        return 1./b
    else:
        return c_square/(t*b)


def mean_update(X, mean, sigma_inv, b, c_square):
    # Returns updated mean

    N, p = X.shape
    mean_hat = np.zeros(p)
    sum_u = 0
    for z in range(N):
        u = u_func(t = np.dot(np.dot(X[z] - mean, sigma_inv), X[z] - mean), b=b, c_square=c_square)
        sum_u += u
        mean_hat += 1./N*u*X[z]

    mean_hat = mean_hat/(1./N*sum_u)

    return mean_hat


def sigma_update(X, mean, sigma_inv, b, c_square):
    # Returns updated sigma

    N, p = X.shape
    sigma_hat = np.zeros((p,p))

    for z in range(N):
        sigma_hat += 1./N*u_func(t = np.dot(np.dot(X[z] - mean, sigma_inv), X[z] - mean), b=b, c_square=c_square)*np.outer(X[z]- mean, X[z]- mean)

    Maronnas_Equation = False
    if Maronnas_Equation:
        sigma_hat=np.dot(np.dot(sigma_hat, sigma_inv), sigma_hat)

    return sigma_hat


def load_data_matlab_csv(k=1):
    # This function is for loading data collected with matlab.
    eng = matlab.engine.start_matlab()
    # Below should take as parameter the path of your matlab code for RSVP Keyboard.
    eng.addpath(eng.genpath('C:\\Users\\Berkan\\Desktop\\GITProjects\\rsvp-keyboard'))
    x, y = eng.python_helper(k, nargout=2)

    return np.array(x), np.array(y)


def demo_m_estimator():
    # Simple demo for m estimators.

    # parser = argparse.ArgumentParser()
    # parser.add_argument('N', help='# of samples', type=int)
    # parser.add_argument('p', help='# of features', type=int)

    # parser.add_argument('mean_mean', help='mean of xi is going to be normal distributed around this value with std: mean_std')
    # parser.add_argument('mean_std', help='How much true mean varies from mean_mean parameter')

    # parser.add_argument('outlier_percentage', help='# = n_outlier/N*100')
    # parser.add_argument('mean_mean_outlier', help='mean of outliers is going to be normal distributed around this value with std: mean_std_outlier')
    # parser.add_argument('mean_std_outlier', help='How much true mean_outlier varies from mean_mean_outlier parameter')
    # parser.add_argument('-i', help='# of iterations', type=int, default = 1000)
    # parser.add_argument('-e', help='epsilon', type=float, default = .1**4)
    # parser.add_argument(-p, help='Plotting std ellipses for 2d case', default = True)

    # args = parser.parse_args()
    #
    # N = args.N
    # p = args.p
    # q = .5
    # true_mean = args.mean_mean + args.mean_std*np.random.standard_normal(p)
    # true_sigma = make_spd_matrix(n_dim=p)
    # N_outlier = int(N*args.outlier_percentage/100)
    # true_mean_outlier = args.mean_mean_outlier + mean_std_outlier*np.random.standard_normal(p)
    # true_sigma_outlier = make_spd_matrix(n_dim=p)
    # iter_num = args.i
    # epsilon = args.e
    # plot_choice = args.p

    # np.random.seed(seed=12)
    N = 150
    p = 75
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

    print 'Sample mean - true mean:\n', sample_mean - true_mean
    print 'Sample cov - true cov:\n', sample_sigma - true_sigma

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
        print s_a_c
        iteration += 1


    print '\n\nFinished in {} iterations.\n\n'.format(iteration)
    print 'M estimate mean - true mean:\n', M_est_mean_new - true_mean
    print 'M estimate sigma - true sigma:\n', M_est_sigma_new - true_sigma
    print '\nΣ|Sample mean - true mean|:', np.sum(np.abs(sample_mean - true_mean))
    print 'Σ|M estimate mean - true mean|:', np.sum(np.abs(M_est_mean_new - true_mean))
    print 'Σ|Sample cov - true cov|:', np.sum(np.sum(np.abs(sample_sigma - true_sigma)))
    print 'Σ|M estimate sigma - true sigma|:', np.sum(np.sum(np.abs(M_est_sigma_new - true_sigma)))

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


def offline_analysis_m(data_folder=None, from_matlab=False, method='regular'):
    """ Train the model by first loading the data and save the trained model.
        Args:
            data_folder(str): Path to data folder
            from_matlab(boolean): If True use csv file from Matlab
        """

    k = 2

    # If data was recorded with Matlab
    if from_matlab:
        x, y = load_data_matlab_csv(k=k)
        y = np.squeeze(y)
        # indices = np.random.permutation(len(y))
        # x = x[:,indices,:]
        # y = y[indices]
    else:
        if not data_folder:
            data_folder = load_experimental_data()

        raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
            data_folder + '/raw_data.csv')

        # TODO: Read from parameters

        dat = sig_pro(raw_dat, fs=fs, k=k)

        # Process triggers.txt
        s_i, t_t_i, t_i = trigger_decoder(mode='calibration',
                                          trigger_loc=data_folder + '/triggers.txt')

        # Channel map can be checked from raw_data.csv file.
        # read_data_csv already removes the timespamp column.
        #                     CM            X3 X2           X1            TRG
        channel_map = [1] * 8 + [0] + [1] * 7 + [0] * 2 + [1] * 2 + [0] + [1] * 3 + [0]
        x, y, num_seq, _ = trial_reshaper(t_t_i, t_i, dat, mode='calibration',
                                          fs=fs, first_sample_time=stamp_time[0], k=k,
                                          channel_map=channel_map)

    if method=='regular':
        model = train_pca_rda_kde_model(x, y, k_folds=10)
    elif method=='m-estimator':
        pass

    print('Saving offline analysis plots!')
    generate_offline_analysis_screen(x, y, model, data_folder)

    print('Saving the model!')
    with open(data_folder + '/model.pkl', 'wb') as output:
        pickle.dump(model, output)
    return model


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    offline_analysis_m(from_matlab=True, method='regular')
    # demo_m_estimator()

