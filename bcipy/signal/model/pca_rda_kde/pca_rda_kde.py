import pickle
from pathlib import Path
from typing import List

import numpy as np
from bcipy.helpers.task import TrialReshaper
from bcipy.signal.exceptions import SignalException
from bcipy.signal.model import ModelEvaluationReport, SignalModel
from bcipy.signal.model.pca_rda_kde.classifier import RegularizedDiscriminantAnalysis
from bcipy.signal.model.pca_rda_kde.cross_validation import (
    cost_cross_validation_auc,
    cross_validation,
)
from bcipy.signal.model.pca_rda_kde.density_estimation import KernelDensityEstimate
from bcipy.signal.model.pca_rda_kde.dimensionality_reduction import (
    ChannelWisePrincipalComponentAnalysis,
)
from bcipy.signal.model.pca_rda_kde.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels


class PcaRdaKdeModel(SignalModel):
    reshaper = TrialReshaper()

    def __init__(self, k_folds: int):
        self.k_folds = k_folds
        self.model = None
        self._ready_to_predict = False

    def fit(self, train_data: np.array, train_labels: np.array) -> SignalModel:
        """
        Train on provided data using K-fold cross validation and return self.

        Parameters:
            train_data: shape (Channels, Trials, Trial_length) preprocessed data
            train_labels: shape (Trials,) binary labels

        Returns:
            trained likelihood model
        """
        model = Pipeline(
            [
                ChannelWisePrincipalComponentAnalysis(var_tol=1e-5, num_ch=train_data.shape[0]),
                RegularizedDiscriminantAnalysis(),
            ]
        )

        # Find the optimal gamma + lambda values
        arg_cv = cross_validation(train_data, train_labels, model=model, k_folds=self.k_folds)

        # Get the AUC using those optimized gamma + lambda
        rda_index = 1  # the index in the pipeline
        model.pipeline[rda_index].lam = arg_cv[0]
        model.pipeline[rda_index].gam = arg_cv[1]
        _, sc_cv, y_cv = cost_cross_validation_auc(
            model, rda_index, train_data, train_labels, arg_cv, k_folds=self.k_folds, split="uniform"
        )

        # After finding cross validation scores do one more round to learn the
        # final RDA model
        model.fit(train_data, train_labels)

        # Insert the density estimates to the model and train using the cross validated
        # scores to avoid over fitting. Observe that these scores are not obtained using
        # the final model
        model.add(KernelDensityEstimate(scores=sc_cv, num_channels=train_data.shape[0]))
        model.pipeline[-1].fit(sc_cv, y_cv)

        self.model = model
        self.prior_class_1 = 0.5
        self.prior_class_0 = 1 - self.prior_class_1
        # self.prior_class_1 = np.sum(train_labels == 1) / len(train_labels)
        # self.prior_class_0 = 1 - self.prior_class_1

        self.classes_ = unique_labels(train_labels)
        self._ready_to_predict = True
        return self

    def evaluate(self, test_data: np.array, test_labels: np.array) -> ModelEvaluationReport:
        """Computes AUROC of the intermediate RDA step of the pipeline using k-fold cross-validation

        Args:
            test_data (np.array): shape (Channels, Trials, Trial_length) preprocessed data.
            test_labels (np.array): shape (Trials,) binary labels.

        Raises:
            SignalException: error if called before model is fit.

        Returns:
            ModelEvaluationReport: stores AUC
        """
        if not self._ready_to_predict:
            raise SignalException("must use model.fit() before model.evaluate()")

        tmp_model = Pipeline([self.model.pipeline[0], self.model.pipeline[1]])

        lam_gam = (self.model.pipeline[1].lam, self.model.pipeline[1].gam)
        tmp, _, _ = cost_cross_validation_auc(
            tmp_model, 1, test_data, test_labels, lam_gam, k_folds=self.k_folds, split="uniform"
        )
        auc = -tmp
        return ModelEvaluationReport(auc)

    def predict(self, data: np.array) -> np.array:
        """
        sklearn-compatible method for predicting
        """
        if not self._ready_to_predict:
            raise SignalException("must use model.fit() before model.predict()")

        # p(l=1 | e) = p(e | l=1) p(l=1)
        probs = self.predict_proba(data)
        return probs.argmax(-1)

    def predict_proba(self, data: np.array) -> np.array:
        """
        sklearn-compatible method for predicting probabilities
        """
        if not self._ready_to_predict:
            raise SignalException("must use model.fit() before model.predict_proba()")

        # p(l=1 | e) = p(e | l=1) p(l=1)
        scores_class_0 = np.exp(self.model.transform(data))[:, 0]
        scores_class_1 = np.exp(self.model.transform(data))[:, 1]
        unnorm_posterior_class_0 = scores_class_0 * self.prior_class_0
        unnorm_posterior_class_1 = scores_class_1 * self.prior_class_1
        posterior_class_0 = unnorm_posterior_class_0 / (unnorm_posterior_class_0 + unnorm_posterior_class_1)
        posterior_class_1 = unnorm_posterior_class_1 / (unnorm_posterior_class_0 + unnorm_posterior_class_1)
        return np.stack([posterior_class_0, posterior_class_1], -1)

    def save(self, path: Path):
        """Save model weights (e.g. after training) to `path`"""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path):
        """Load pretrained model weights from `path`"""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._ready_to_predict = True
