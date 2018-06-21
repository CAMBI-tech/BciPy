from bcipy.signal_model.mach_learning.dimensionality_reduction.function_dim_reduction \
    import ChannelWisePrincipalComponentAnalysis, MPCA

from bcipy.signal_model.mach_learning.classifier.function_classifier \
    import RegularizedDiscriminantAnalysis, MDiscriminantAnalysis

from bcipy.signal_model.mach_learning.generative_mods.function_density_estimation import KernelDensityEstimate

from bcipy.signal_model.mach_learning.pipeline import Pipeline
from bcipy.signal_model.mach_learning.cross_valid import *

from sklearn import metrics
from scipy.stats import iqr


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
    pca = ChannelWisePrincipalComponentAnalysis(var_tol=.1**5,
                                                num_ch=x.shape[0])
    rda = RegularizedDiscriminantAnalysis()

    model = Pipeline()
    model.add(pca)
    model.add(rda)

    # Get the AUC before the regularization
    scores = model.fit_transform(x, y)
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=1)
    auc_init = metrics.auc(fpr, tpr)

    # Cross validate
    lam, gam = cross_validate_parameters(x, y, model=model)

    print('Optimized val [gam:{} \ lam:{}]'.format(gam, lam))
    model.pipeline[1].lam = lam
    model.pipeline[1].gam = gam

    # Insert the density estimates to the model and train
    bandwidth = 1.06 * min(
        np.std(scores), iqr(scores) / 1.34) * np.power(x.shape[0], -0.2)
    model.add(KernelDensityEstimate(bandwidth=bandwidth))
    model.fit(x, y)

    # Report AUC
    print('AUC-i: {}, AUC-cv: {}'.format(auc_init, model.last_cv_auc))

    return model


def train_m_estimator_pipeline(x, y):
    """ Trains Cw-m-PCA m-RDA KDE model given the input data and labels, returns the model
        Args:
            x(ndarray[float]): C x N x p data array
            y(ndarray[int]): N x 1 observation (class) array
                N is number of samples p is dimensionality of features
                C is number of channels
            k_folds(int): Number of cross validation folds
            cross_validate(Boolean): True if cross validation AUC is to be calculated. Increases required time.
        Return:
            model(pipeline): trained model
    """

    # Train PCA and MDA to cross validate hyper parameters
    model = Pipeline()

    pca = MPCA(var_tol=.1**5)
    mda = MDiscriminantAnalysis()

    model.add(pca)
    model.add(mda)

    lam, gam = cross_validate_parameters(x=x, y=y, model=model)
    cv_auc = model.last_cv_auc
    print('Optimized val [lam:{} \ gam:{}]'.format(lam, gam))

    # Train the model with optimized hyper parameters to find once again the CV AUC
    model = Pipeline()

    pca = MPCA(var_tol=.1**5)
    mda = MDiscriminantAnalysis()

    model.add(pca)
    model.add(mda)

    model.pipeline[1].lam = lam
    model.pipeline[1].gam = gam

    scores = model.fit_transform(x, y)
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=1)
    auc_all = metrics.auc(fpr, tpr)

    bandwidth = 1.06 * min(np.std(scores), iqr(scores) / 1.34) * np.power(x.shape[0], -0.2)

    model = Pipeline()

    pca = MPCA(var_tol=.1**5)
    mda = MDiscriminantAnalysis()

    model.add(pca)
    model.add(mda)
    model.add(KernelDensityEstimate(bandwidth=bandwidth))

    model.pipeline[1].lam = lam
    model.pipeline[1].gam = gam
    model.fit(x, y)
    model.last_cv_auc = cv_auc

    # Report AUC
    print('AUC-complete_data: {}, AUC-cv: {}'.format(auc_all, model.last_cv_auc))

    return model
