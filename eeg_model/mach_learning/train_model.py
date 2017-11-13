import numpy as np
from eeg_model.mach_learning.classifier.function_classifier \
    import RegularizedDiscriminantAnalysis
from eeg_model.mach_learning.dimensionality_reduction.function_dim_reduction \
    import ChannelWisePrincipalComponentAnalysis
from eeg_model.mach_learning.wrapper import PipeLine
from eeg_model.mach_learning.cross_validation import cross_validation
from eeg_model.mach_learning.generative_mods.function_density_estimation \
    import KernelDensityEstimate
from sklearn import metrics
from scipy.stats import iqr


# TODO: Horrible implementation make it modular!
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
    pca = ChannelWisePrincipalComponentAnalysis(tol=0)
    model = PipeLine()
    model.add(pca)
    model.add(rda)

    # Get the AUC before the regularization
    sc = model.fit_transform(x, y)
    fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
    auc_init = metrics.auc(fpr, tpr)

    # Cross validate
    arg_cv = cross_validation(x, y, model=model, k_folds=k_folds)
    auc_cv = arg_cv[2]

    idx_max_auc = np.where(auc_cv == np.max(auc_cv))[0][0]
    lam = arg_cv[0][idx_max_auc]
    gam = arg_cv[1][idx_max_auc]
    print('Optimized val [gam:{} \ lam:{}]'.format(lam, gam))
    model.pipeline[1].lam = lam
    model.pipeline[1].gam = gam
    sc = model.fit_transform(x, y)

    # Insert the density estimates to the model and train
    bandwidth = 1.06 * min(
        np.std(sc), iqr(sc) / 1.34) * np.power(x.shape[0], -0.2)
    model.add(KernelDensityEstimate(bandwidth=bandwidth))
    model.fit(x, y)

    # Report AUC
    print('AUC-i: {}, AUC-cv: {}'.format(auc_init, np.max(np.mean(auc_cv))))

    return model
