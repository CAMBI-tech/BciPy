import numpy as np
from bcipy.signal.model.mach_learning.classifier.function_classifier \
    import RegularizedDiscriminantAnalysis
from bcipy.signal.model.mach_learning.dimensionality_reduction.function_dim_reduction \
    import ChannelWisePrincipalComponentAnalysis
from bcipy.signal.model.mach_learning.pipeline import Pipeline
from bcipy.signal.model.mach_learning.cross_validation import cross_validation, \
    cost_cross_validation_auc
from bcipy.signal.model.mach_learning.generative_mods.function_density_estimation \
    import KernelDensityEstimate
from sklearn import metrics
from scipy.stats import iqr

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s', )


def train_pca_rda_kde_model(x, y, k_folds=10):
    """ Trains the Cw-PCA RDA KDE model given the input data and labels with
        cross validation and returns the model
        Args:
            x(ndarray[float]): C x N x k data array
            y(ndarray[int]): N x 1 observation (class) array
                N is number of samples k is dimensionality of features
                C is number of channels
            k_folds(int): Number of cross validation folds
        Return:
            model(pipeline): trained likelihood model
            """

    # Pipeline is the model. It can be populated manually
    rda = RegularizedDiscriminantAnalysis()
    pca = ChannelWisePrincipalComponentAnalysis(var_tol=.1 ** 5,
                                                num_ch=x.shape[0])
    model = Pipeline()
    model.add(pca)
    model.add(rda)

    # Cross validate
    arg_cv = cross_validation(x, y, model=model, k_folds=k_folds)

    # Get the AUC before the regularization
    tmp, sc_cv, y_cv = cost_cross_validation_auc(model, 1, x, y, arg_cv,
                                                 k_folds=10, split='uniform')
    auc_init = -tmp
    # Start Cross validation
    lam = arg_cv[0]
    gam = arg_cv[1]
    logging.debug('Optimized val [gam:{} \ lam:{}]'.format(lam, gam))
    model.pipeline[1].lam = lam
    model.pipeline[1].gam = gam
    tmp, sc_cv, y_cv = cost_cross_validation_auc(model, 1, x, y, arg_cv,
                                                 k_folds=10, split='uniform')
    auc_cv = -tmp

    # After finding cross validation scores do one more round to learn the final RDA model
    model.fit(x, y)

    # Insert the density estimates to the model and train using the cross validated
    # scores to avoid over fitting. Observe that these scores are not obtained using
    # the final model
    bandwidth = 1.06 * min(
        np.std(sc_cv), iqr(sc_cv) / 1.34) * np.power(x.shape[0], -0.2)
    model.add(KernelDensityEstimate(bandwidth=bandwidth))
    model.pipeline[-1].fit(sc_cv, y_cv)

    # Report AUC
    logging.debug('AUC-i: {}, AUC-cv: {}'.format(auc_init, auc_cv))

    return model, auc_cv
