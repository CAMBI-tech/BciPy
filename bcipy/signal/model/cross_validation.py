import numpy as np

import scipy.optimize
from sklearn import metrics

import logging

log = logging.getLogger(__name__)


def cost_cross_validation_auc(model, opt_el, x, y, param, k_folds=10,
                              split='uniform'):
    """ Minimize cost of the overall -AUC.
        Cost function: given a particular architecture (model). Fits the
        parameters to the folds with leave one fold out procedure. Calculates
        scores for the validation fold. Concatenates all calculated scores
        together and returns a -AUC vale.
        Args:
            model(pipeline): model to be iterated on
            opt_el(int): number of the element in pipeline to be optimized
            x(ndarray[float]): N x k data array
            y(ndarray[int]): N x k observation (class) array
                N is number of samples k is dimensionality of features
            param(list[float]): ordered hyper parameter list for the
                regularization
            k_folds(int): number of folds
            split(string): split type,
                'uniform': Takes the data as is
        Return:
            -auc(float): negative AUC value for current setup
            sc_h(ndarray[float]): scores computed for each validation fold
            y_valid_h(ndarray[int]): labels of the scores for each validation fold
                y_valid_h[i] is basically the label for sc_h[i] """

    num_samples = x.shape[1]
    fold_len = np.floor(float(num_samples) / k_folds)

    model.pipeline[opt_el].lam = param[0]
    model.pipeline[opt_el].gam = param[1]

    fold_x, fold_y = [], []
    sc_h, y_valid_h = [], []
    if split == 'uniform':
        for idx_fold in range(k_folds):
            fold_x.append(x[:, int(idx_fold * fold_len):int((idx_fold + 1) * fold_len), :])
            fold_y.append(y[int(idx_fold * fold_len):int((idx_fold + 1) * fold_len)])

            if len(np.unique(fold_y[idx_fold])) != 2:
                raise Exception(
                    f'Cannot use {split}-folding in cross_validation '
                    'or # of folds is inconsistent')

        for idx_fold in range(k_folds):
            list_valid = idx_fold
            list_train = list(set(range(k_folds)) - set([idx_fold]))

            x_train = np.concatenate([fold_x[i] for i in list_train], axis=1)
            y_train = np.concatenate([fold_y[i] for i in list_train], axis=0)
            x_valid = fold_x[list_valid]
            y_valid = fold_y[list_valid]

            model.fit(x_train, y_train)
            sc = model.transform(x_valid)

            sc_h.append(sc)
            y_valid_h.append(y_valid)

    y_valid_h = np.concatenate(np.array(y_valid_h))
    sc_h = np.concatenate(np.array(sc_h))
    fpr, tpr, _ = metrics.roc_curve(y_valid_h, sc_h, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return -auc, sc_h, y_valid_h


def nonlinear_opt(model, opt_el, x, y, init=None, op_type='cost_auc',
                  arg_op_type=[10, 'uniform']):
    """ Optimizes lambda, gamma values for given  penalty function
        Args:
            model(pipeline): model to be iterated on
            opt_el(int): number of the element in pipeline to be optimized
            x(ndarray[float]): N x k data array
            y(ndarray[int]): N x k observation (class) array
                 N is number of samples k is dimensionality of features
            init(list[float]): initial values for gamma and lambda
            op_type(string): type of the optimization
        Return:
            arg_opt(list[float]): optimized hyper parameters
            """

    # Get initial values
    if not init:
        init = [model.pipeline[opt_el].lam, model.pipeline[opt_el].gam]
    if op_type:
        # TODO: maybe we should not have such an option and set it by ourselves
        if op_type == 'cost_auc':
            k_folds, split = arg_op_type

            def cost_fun_param(b):
                return cost_cross_validation_auc(model, opt_el, x, y, [b[0], b[1]],
                                                 k_folds=k_folds, split=split)[0]

        # Intervals for lambda and gamma parameters
        # Observe that 0 < lam < 1, 0 < gam < 1
        def cst_1(v):

            return v[0] - np.power(0.1, 15)

        def cst_2(v):

            return v[1] - np.power(0.1, 15)

        def cst_3(v):

            return 1 - v[0]

        def cst_4(v):

            return 1 - v[1]

        arg_opt = scipy.optimize.fmin_cobyla(cost_fun_param, x0=init,
                                             disp=False,
                                             cons=[cst_1, cst_2, cst_3,
                                                   cst_4])
    return arg_opt


def cross_validation(x, y, model, opt_el=1, k_folds=10, split='uniform'):
    """ Cross validation function for hyper parameter optimization
        Args:
            x(ndarray[float]): C x N x k data array
            y(ndarray[int]): N x 1 observation (class) array
                N is number of samples k is dimensionality of features
                C is number of channels
            model(pipeline): model to be optimized
            opt_el(int): element in the model to update hyper-params in [0,M]
            k_folds(int): number of folds
            split(string): split type,
                'uniform': Takes the data as is
        Return:
            return(list[list[float],list[float],list[float]]): lists of lambda,
                gamma and AUC values for each fold respectively.
            """

    log.debug('Starting Cross Validation !')
    arg_opt = nonlinear_opt(model, opt_el, x, y, op_type='cost_auc',
                            arg_op_type=[k_folds, split])
    return arg_opt
