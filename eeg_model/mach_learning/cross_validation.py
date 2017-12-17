import numpy as np
import scipy.optimize
from sklearn import metrics
from utils.progress_bar import progress_bar


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
            -auc(float): negative AUC value for current setup """

    num_samples = x.shape[1]
    fold_len = np.floor(float(num_samples) / k_folds)

    model.pipeline[1].lam = param[0]
    model.pipeline[1].gam = param[1]

    fold_x, fold_y = [], []
    sc, y_sc = [], []
    if split == 'uniform':
        for idx_fold in range(k_folds + 1):
            fold_x.append(x[:, int(idx_fold * fold_len):int(
                (idx_fold + 1) * fold_len), :])
            fold_y.append(y[int(idx_fold * fold_len):int((idx_fold + 1) *
                                                         fold_len)])
            if len(np.unique(fold_y[idx_fold])) == 1:
                raise Exception('Cannot use {}-folding in cross_validation '
                                'or # of folds is inconsistent'.format(split))

        for idx_fold in range(k_folds):
            list_valid = idx_fold
            list_train = list(set(range(k_folds)) - set([idx_fold]))

            x_train = np.concatenate([fold_x[i] for i in list_train], axis=1)
            y_train = np.concatenate([fold_y[i] for i in list_train], axis=0)
            x_valid = fold_x[list_valid]
            y_valid = fold_y[list_valid]

            model.fit(x_train, y_train)
            sc.append(model.transform(x_valid))
            y_sc.append(y_valid)

    sc = np.concatenate(sc)
    y_sc = np.concatenate(y_sc)
    fpr, tpr, _ = metrics.roc_curve(y_sc, sc, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return -auc


def grid_search(model, opt_el, x, y, grid=[5, 5], op_type='cost_auc',
                arg_op_type=[10, 'uniform']):
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
        k_folds, split = arg_op_type
        # This specifies the different candidate values we want to try.
        # The grid search will try all combination of these parameter values
        # and select the set of parameters that provides the most accurate
        # model.
        param_cand = {  # dictionary of parameter candidates
            'lam': np.linspace(.01, .99, grid[0], endpoint=False),
            'gam': np.linspace(.01, .99, grid[1], endpoint=False),
        }
        best_auc = 0  # auc can't be smaller than 0
        arg_opt = {'lam': 0, 'gam': 0}

        # For every coordinate on the grid, try every combination of
        # hyper parameters:
        progress_bar(0, grid[0] * grid[1], prefix='Progress:',
                     suffix='Complete', length=50)
        tmp_counter = 0
        for i in range(len(param_cand['lam'])):
            for j in range(len(param_cand['gam'])):
                auc = -cost_cross_validation_auc(model, opt_el, x, y,
                                                 [param_cand['lam'][i],
                                                  param_cand['gam'][j]],
                                                 k_folds=k_folds, split=split)
                if auc > best_auc:
                    best_auc = auc
                    arg_opt['lam'], arg_opt['gam'] = param_cand['lam'][i], \
                                                     param_cand['gam'][j]

                tmp_counter += 1
                progress_bar(tmp_counter, grid[0] * grid[1],
                             prefix='Progress:', suffix='Complete', length=50)
    else:
        # TODO: Handle this case
        print('Error: Operation type other than AUC cost.')

    # This returns the parameter estimates with the highest scores:
    return [arg_opt['lam'], arg_opt['gam']]


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
            cost_fun_param = lambda b: cost_cross_validation_auc(
                model, opt_el, x, y, [b[0], b[1]], k_folds=k_folds,
                split=split)

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

    print('Starting Cross Validation !')
    arg_opt = nonlinear_opt(model, opt_el, x, y, op_type='cost_auc',
                            arg_op_type=[k_folds, split])
    return arg_opt
