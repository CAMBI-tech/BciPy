import numpy as np
import scipy.optimize
from sklearn import metrics


def cost_auc(model, opt_el, x, y, lam, gam):
    """ Minimize cost of the overall -AUC
        Args:
            model(pipeline): model to be iterated on
            opt_el(int): number of the element in pipeline to be optimized
            x(ndarray[float]): N x k data array
            y(ndarray[int]): N x k observation (class) array
                N is number of samples k is dimensionality of features
            lam(float): cost function lambda to iterate over
            gam(float): cost function gamma to iterate over
        Return:
            -auc(float): negative AUC value for current setup
            """

    # x1, x2, y1, y2 = train_test_split(x, y, test_size=0.1)
    # self.fit_param(x1, y1, self.prior)
    model.pipeline[opt_el].regularize(lam=lam, gam=gam)
    sc = model.transform(x)
    fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return -auc


def param_nonlinear_opt(model, opt_el, x, y, init=None, op_type='cost_auc'):
    """ Optimizes lambda, gamma values for given  penalty function
        Args:
            model(pipeline): model to be iterated on
            opt_el(int): number of the element in pipeline to be optimized
            x(ndarray[float]): N x k data array
            y(ndarray[int]): N x k observation (class) array
                 N is number of samples k is dimensionality of features
            init(list[float]): initial values for gamma and lambda
            op_type(string): type of the optimization
            """

    # Get initial values
    if not init:
        init = [model.pipeline[opt_el].lam, model.pipeline[opt_el].gam]
    if op_type:
        # TODO: maybe we should not have such an option and set it by ourselves
        if op_type == 'cost_auc':
            cost_fun_param = lambda b: cost_auc(
                model, opt_el, x, y, b[0], b[1])

        # Intervals for lambda and gamma parameters
        # Observe that 0 < lam < 1, 0 < gam < 1
        cst_1 = lambda v: v[0]
        cst_2 = lambda v: v[1]
        cst_3 = lambda v: 1 - v[0]
        cst_4 = lambda v: 1 - v[1]

        arg_opt = scipy.optimize.fmin_cobyla(cost_fun_param, x0=init,
                                             disp=False,
                                             cons=[cst_1, cst_2, cst_3,
                                                   cst_4])
    return arg_opt

    # TODO: Insert cost functions for parameter update below!


def cross_validation(x, y, model, opt_el=1, K=10, split='uniform'):
    """ Cross validation object implementation which directly follows the
        MATLAB implementation
        Args:
            x(ndarray[float]): C x N x k data array
            y(ndarray[int]): N x 1 observation (class) array
                N is number of samples k is dimensionality of features
                C is number of channels
            model(pipeline): model to be optimized
            opt_el(int): element in the model to update hyper-params in [0,M]
            K(int): number of folds
            split(string): split type,
                'uniform': Takes the data as is
            """
    num_samples = x.shape[1]
    fold_len = np.floor(float(num_samples) / K)

    fold_x, fold_y = [], []
    if split == 'uniform':
        for idx_fold in range(K):
            fold_x.append(x[:, int(idx_fold * fold_len):int(
                (idx_fold + 1) * fold_len), :])
            fold_y.append(y[int(idx_fold * fold_len):int((idx_fold + 1) *
                                                         fold_len)])

    # Split data
    lam, gam, auc_h = [], [], []
    for idx_fold in range(K):
        list_valid = idx_fold
        list_train = list(set(range(K)) - set([idx_fold]))

        x_train = np.concatenate([fold_x[i] for i in list_train], axis=1)
        y_train = np.concatenate([fold_y[i] for i in list_train], axis=0)
        x_valid = fold_x[list_valid]
        y_valid = fold_y[list_valid]

        model.fit(x_train, y_train)
        arg_opt = param_nonlinear_opt(model, opt_el, x_valid, y_valid)
        lam.append(arg_opt[0])
        gam.append(arg_opt[1])

        sc = model.transform(x)
        fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
        auc_h.append(metrics.auc(fpr, tpr))

    lam = np.asarray(lam)
    gam = np.asarray(gam)
    auc_h = np.asarray(auc_h)

    return [lam, gam, auc_h]
