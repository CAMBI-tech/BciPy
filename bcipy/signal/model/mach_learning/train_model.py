import numpy as np
from bcipy.signal.model.mach_learning.classifier.function_classifier \
    import RegularizedDiscriminantAnalysis
from bcipy.signal.model.mach_learning.dimensionality_reduction.function_dim_reduction \
    import ChannelWisePrincipalComponentAnalysis
from bcipy.signal.model.mach_learning.pipeline import Pipeline
from bcipy.signal.model.mach_learning.cross_validation import cross_validation, cost_cross_validation_auc
from bcipy.signal.model.mach_learning.generative_mods.function_density_estimation \
    import KernelDensityEstimate
from sklearn import metrics
from scipy.stats import iqr

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


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
    pca = ChannelWisePrincipalComponentAnalysis(var_tol=.1**5,
                                                num_ch=x.shape[0])
    model = Pipeline()
    model.add(pca)
    model.add(rda)

    # Get the AUC before the regularization
    sc = model.fit_transform(x, y)
    fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
    auc_init = metrics.auc(fpr, tpr)

    # Cross validate
    arg_cv = cross_validation(x, y, model=model, k_folds=k_folds)

    lam = arg_cv[0]
    gam = arg_cv[1]
    logging.debug('Optimized val [gam:{} \ lam:{}]'.format(lam, gam))
    model.pipeline[1].lam = lam
    model.pipeline[1].gam = gam
    auc_cv = -cost_cross_validation_auc(model, 1, x, y, arg_cv,
                                        k_folds=10, split='uniform')

    # Insert the density estimates to the model and train
    bandwidth = 1.06 * min(
        np.std(sc), iqr(sc) / 1.34) * np.power(x.shape[0], -0.2)
    model.add(KernelDensityEstimate(bandwidth=bandwidth))
    model.fit(x, y)

    # Report AUC
    logging.debug('AUC-i: {}, AUC-cv: {}'.format(auc_init, auc_cv))

    return model, auc_cv
