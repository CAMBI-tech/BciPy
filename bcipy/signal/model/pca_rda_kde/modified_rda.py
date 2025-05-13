import numpy as np

from bcipy.signal.model import SignalModel
from bcipy.signal.model.classifier import RegularizedDiscriminantAnalysis
from bcipy.signal.model.cross_validation import (cost_cross_validation_auc, cross_validation)
from bcipy.signal.model.pipeline import Pipeline
from bcipy.signal.model.dimensionality_reduction import ChannelWisePrincipalComponentAnalysis
from bcipy.signal.model.identity import Identity


class ModifiedRdaModel():

    def __init__(self, k_folds: int = 10, data_type: str = "synthetic", pca_n_components: float = 0.9, n_classes: int = 2):
        self.k_folds = k_folds
        # min and max values for the likelihood ratio output
        self.min = 1e-2
        self.max = 1e2
        self.model = None
        self.auc = None
        self.data_type = data_type
        self.pca_n_components = pca_n_components
        self.optimization_elements = 1
        self.n_classes = n_classes

    @property
    def ready_to_predict(self) -> bool:
        """Returns True if a model has been trained"""
        return bool(self.model)

    def fit(self, train_data: np.array, train_labels: np.array) -> SignalModel:
        """
        Train on provided data using K-fold cross validation and return self.

        Parameters:
            train_data: shape (Channels, Trials, Trial_length) preprocessed data
            train_labels: shape (Trials,) binary labels

        Returns:
            trained likelihood model
        """
        if self.data_type == "synthetic":
            model = Pipeline(
                [
                    Identity(n_components=self.pca_n_components, num_ch=train_data.shape[0]),
                    RegularizedDiscriminantAnalysis()
                ]
            )
            self.optimization_elements = 1
            
        else:
            model = Pipeline(
                [
                    ChannelWisePrincipalComponentAnalysis(n_components=self.pca_n_components, num_ch=train_data.shape[0]),
                    RegularizedDiscriminantAnalysis(),
                ]
            )
            self.optimization_elements = 1
        # Find the optimal gamma + lambda values
        arg_cv = cross_validation(train_data, train_labels, model=model, opt_el=self.optimization_elements, 
                                  k_folds=self.k_folds, n_classes=self.n_classes)

        # Get the AUC using those optimized gamma + lambda
        rda_index = 1  # the index in the pipeline
        model.pipeline[rda_index].lam = arg_cv[0]
        model.pipeline[rda_index].gam = arg_cv[1]
        tmp, sc_cv, y_cv = cost_cross_validation_auc(
            model, rda_index, train_data, train_labels, arg_cv, k_folds=self.k_folds, split="uniform", n_classes=self.n_classes)
        self.auc = -tmp
        # After finding cross validation scores do one more round to learn the
        # final RDA model
        
        model.fit(train_data, train_labels)

        self.model = model

        return model.pipeline[rda_index].mean_i, model.pipeline[rda_index].cov_i
    
