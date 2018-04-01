from signal_model.mach_learning.dimensionality_reduction.function_dim_reduction \
    import ChannelWisePrincipalComponentAnalysis, MPCA
from signal_model.mach_learning.classifier.function_classifier \
    import RegularizedDiscriminantAnalysis
from signal_model.mach_learning.generative_mods.function_density_estimation import KernelDensityEstimate
from signal_model.mach_learning.pipeline import Pipeline
from signal_model.mach_learning.cross_valid import *
from sklearn import metrics
from scipy.stats import iqr


def train_pca_rda_kde_model(x, x_artifact, y, k_folds=10):
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
    sc = model.fit_transform(x_artifact, y)
    fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
    auc_init = metrics.auc(fpr, tpr)

    # Cross validate
    lam, gam = cross_validate_parameters(x_artifact, y, model=model, k_folds=k_folds)

    print('Optimized val [gam:{} \ lam:{}]'.format(gam, lam))
    model.pipeline[1].lam = lam
    model.pipeline[1].gam = gam
    auc_cv = cross_validate_model(x=x, x_artifact=x_artifact, y=y, model=model)

    # Insert the density estimates to the model and train
    bandwidth = 1.06 * min(
        np.std(sc), iqr(sc) / 1.34) * np.power(x.shape[0], -0.2)
    model.add(KernelDensityEstimate(bandwidth=bandwidth))
    model.fit(x, y)

    # Report AUC
    print('AUC-i: {}, AUC-cv: {}'.format(auc_init, auc_cv))

    return model, auc_cv


def train_m_estimator_pipeline(x, x_artifact, y, k_folds=10):
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

    pca = MPCA(var_tol=.1**5, output_type='RDA', n_folds=k_folds)
    rda = RegularizedDiscriminantAnalysis()

    model = Pipeline()
    model.add(pca)
    model.add(rda)

    # Find the cross validation AUC for the model.
    lam, gam = cross_validate_parameters(x=x_artifact, y=y, model=model, k_folds=k_folds)
    print('Optimized val [gam:{} \ lam:{}]'.format(gam, lam))

    pca = MPCA(var_tol=.1**5, output_type='RDA', n_folds=k_folds)
    rda = RegularizedDiscriminantAnalysis()

    model = Pipeline()
    model.add(pca)
    model.add(rda)

    model.pipeline[1].lam = lam
    model.pipeline[1].gam = gam
    auc_cv = cross_validate_model(x=x, x_artifact=x_artifact, y=y, model=model)

    # # Now train on complete data for the final model
    # sc = model.fit_transform(x_artifact, y)
    # fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
    # auc_all = metrics.auc(fpr, tpr)
    #
    # # Insert the density estimates to the model and train
    # bandwidth = 1.06 * min(
    #     np.std(sc), iqr(sc) / 1.34) * np.power(x.shape[0], -0.2)
    # model.add(KernelDensityEstimate(bandwidth=bandwidth))
    # model.fit(x, y)

    # Report AUC
    print('AUC-complete_data: {}, AUC-cv: {}'.format(1, auc_cv))

    return model, auc_cv
