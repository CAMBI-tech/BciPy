import numpy as np
import scipy.optimize
from sklearn import metrics
import time
import warnings
from utils.progress_bar import progress_bar


def cost_auc(model, opt_el, x, y, param):
    """ Minimize cost of the overall -AUC
        Args:
            model(pipeline): model to be iterated on
            opt_el(int): number of the element in pipeline to be optimized
            x(ndarray[float]): N x k data array
            y(ndarray[int]): N x k observation (class) array
                N is number of samples k is dimensionality of features
            param(list[float]): ordered hyper parameter list for the
                regularization
        Return:
            -auc(float): negative AUC value for current setup
            """

    # x1, x2, y1, y2 = train_test_split(x, y, test_size=0.1)
    # self.fit_param(x1, y1, self.prior)
    model.pipeline[opt_el].regularize(param)
    sc = model.transform(x)
    fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return -auc


def grid_search(model, opt_el, x, y, grid=[10, 10], op_type='cost_auc'):
    """ Description: This function performs an exhaustive grid search
                     to estimate the hyper parameters lambda and gamma
                     that minimize the cost of AUC.
        Args:
            model(pipeline): model to be iterated on
            opt_el(int): number of the element in pipeline to be optimized
            x(ndarray[float]): N x k data array
            y(ndarray[int]): N x k observation (class) array
                 N is number of samples k is dimensionality of features
            grid(list(int)): a list of 2 numbers for grid
            op_type(string): type of the optimization
        Returns:
            arg_opt(list[float]): optimized hyper parameters
            """
    if op_type == 'cost_auc':
        # This specifies the different candidate values we want to try.
        # The grid search will try all combination of these parameter values
        # and select the set of parameters that provides the most accurate
        # model.
        param_cand = {  # dictionary of parameter candidates
            'lam': np.linspace(0, 1, grid[0], endpoint=False),
            'gam': np.linspace(0, 1, grid[1], endpoint=False),
        }
        best_auc = 1  # auc can't be bigger than 1
        arg_opt = {'lam': 0, 'gam': 0}

        # For every coordinate on the grid, try every combination of
        # hyper parameters:
        for i in range(len(param_cand['lam'])):
            for j in range(len(param_cand['gam'])):
                auc = cost_auc(model, opt_el, x, y, [param_cand['lam'][i],
                                                     param_cand['gam'][j]])
                if auc < best_auc:
                    best_auc = auc
                    arg_opt['lam'], arg_opt['gam'] = param_cand['lam'][i], \
                                                     param_cand['gam'][j]
    else:
        # TO DO, Handle this case
        print('Error: Operation type other than AUC cost.')

    # This returns the parameter estimates with the highest scores:
    return [arg_opt['lam'], arg_opt['gam']]


def nonlinear_opt(model, opt_el, x, y, init=None, op_type='cost_auc'):
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
            cost_fun_param = lambda b: cost_auc(
                model, opt_el, x, y, [b[0], b[1]])

        # Intervals for lambda and gamma parameters
        # Observe that 0 < lam < 1, 0 < gam < 1
        cst_1 = lambda v: v[0] - np.power(0.1, 15)
        cst_2 = lambda v: v[1] - np.power(0.1, 15)
        cst_3 = lambda v: 1 - v[0]
        cst_4 = lambda v: 1 - v[1]

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
    num_samples = x.shape[1]
    fold_len = np.floor(float(num_samples) / k_folds)

    fold_x, fold_y = [], []
    if split == 'uniform':
        for idx_fold in range(k_folds):
            fold_x.append(x[:, int(idx_fold * fold_len):int(
                (idx_fold + 1) * fold_len), :])
            fold_y.append(y[int(idx_fold * fold_len):int((idx_fold + 1) *
                                                         fold_len)])
            if len(np.unique(fold_y[idx_fold])) == 1:
                raise Exception('Cannot use {}-folding in cross_validation '
                                'or # of folds is inconsistent'.format(split))

    # Split data
    lam, gam, auc_h = [], [], []
    print('Starting Cross Validation !')
    progress_bar(0, k_folds, prefix='Progress:', suffix='Complete', length=50)
    t = time.time()
    for idx_fold in range(k_folds):
        progress_bar(idx_fold + 1, k_folds, prefix='Progress:',
                     suffix='Complete', length=50)
        list_valid = idx_fold
        list_train = list(set(range(k_folds)) - set([idx_fold]))

        x_train = np.concatenate([fold_x[i] for i in list_train], axis=1)
        y_train = np.concatenate([fold_y[i] for i in list_train], axis=0)
        x_valid = fold_x[list_valid]
        y_valid = fold_y[list_valid]

        model.fit(x_train, y_train)
        arg_opt = grid_search(model, opt_el, x_valid, y_valid)
        lam.append(arg_opt[0])
        gam.append(arg_opt[1])
        model.pipeline[opt_el].regularize(arg_opt)

        sc = model.transform(x)
        fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
        auc_h.append(metrics.auc(fpr, tpr))

    print('Cross Validation Elapsed in {0:.2f}[s]!'.format((time.time() - t)))

    lam = np.asarray(lam)
    gam = np.asarray(gam)
    auc_h = np.asarray(auc_h)

    return [lam, gam, auc_h]
