import numpy as np
from bcipy.signal.model.ml.classifier import RegularizedDiscriminantAnalysis
from bcipy.signal.model.ml.dimensionality_reduction import ChannelWisePrincipalComponentAnalysis
from bcipy.signal.model.ml.pipeline import Pipeline
from bcipy.signal.model.ml.cross_validation import cross_validation, cost_cross_validation_auc
from bcipy.signal.model.ml.density_estimation import KernelDensityEstimate
from scipy.stats import iqr

import logging

log = logging.getLogger(__name__)


def train_pca_rda_kde_model(x, y, k_folds=10):
    """Trains the Cw-PCA RDA KDE model given the input data and labels with
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
    # TODO - goes in SignalModel.__init__()
    rda = RegularizedDiscriminantAnalysis()
    pca = ChannelWisePrincipalComponentAnalysis(var_tol=0.1 ** 5, num_ch=x.shape[0])
    model = Pipeline()
    model.add(pca)
    model.add(rda)

    # Get the AUC using default gamma + lambda values
    arg_cv = [model.pipeline[-1].lam, model.pipeline[-1].gam]
    tmp, sc_cv, y_cv = cost_cross_validation_auc(model, 1, x, y, arg_cv, k_folds=10, split="uniform")
    auc_init = -tmp

    # Now find the optimal gamma + lambda values
    # NOTE - previously, auc_init == auc_cv always. Now, auc_init should be worse
    # (because it is measured before gamma and lambda are optimized)
    arg_cv = cross_validation(x, y, model=model, k_folds=k_folds)
    lam = arg_cv[0]
    gam = arg_cv[1]
    log.debug(r"Optimized val [gam:{} \ lam:{}]".format(lam, gam))

    # Get the AUC using those optimized gamma + lambda
    model.pipeline[1].lam = lam
    model.pipeline[1].gam = gam
    tmp, sc_cv, y_cv = cost_cross_validation_auc(model, 1, x, y, arg_cv, k_folds=10, split="uniform")
    auc_cv = -tmp

    # After finding cross validation scores do one more round to learn the
    # final RDA model
    model.fit(x, y)

    # Insert the density estimates to the model and train using the cross validated
    # scores to avoid over fitting. Observe that these scores are not obtained using
    # the final model
    bandwidth = 1.06 * min(np.std(sc_cv), iqr(sc_cv) / 1.34) * np.power(x.shape[0], -0.2)
    model.add(KernelDensityEstimate(bandwidth=bandwidth))
    model.pipeline[-1].fit(sc_cv, y_cv)

    # Report AUC
    log.debug("AUC-i: {}, AUC-cv: {}".format(auc_init, auc_cv))

    return model, auc_cv
