import pickle
from pathlib import Path
from typing import List

import numpy as np
from bcipy.signal.model.base_model import ModelEvaluationReport, SignalModel
from bcipy.signal.model.pca_rda_kde.classifier import RegularizedDiscriminantAnalysis
from bcipy.signal.model.pca_rda_kde.cross_validation import cost_cross_validation_auc, cross_validation
from bcipy.signal.model.pca_rda_kde.density_estimation import KernelDensityEstimate
from bcipy.signal.model.pca_rda_kde.dimensionality_reduction import ChannelWisePrincipalComponentAnalysis
from bcipy.signal.model.pca_rda_kde.pipeline import Pipeline
from bcipy.signal.exceptions import SignalException


class PcaRdaKdeModel(SignalModel):
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
        model = Pipeline([ChannelWisePrincipalComponentAnalysis(var_tol=1e-5, num_ch=train_data.shape[0]),
                          RegularizedDiscriminantAnalysis()])

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

    def predict(self, data: np.array, inquiry: List[str], symbol_set: List[str]) -> np.array:
        """
        For each trial in `data`, compute a likelihood ratio to update that symbol's probability.
        Rather than just computing an update p(e|l=+) for the seen symbol and p(e|l=-) for all unseen symbols,
        we compute a likelihood ratio p(e | l=+) / p(e | l=-) to update the seen symbol, and all other symbols
        can receive a multiplicative update of 1.

        Args:
            data (np.array): EEG data with shape (n_channel, n_trial, n_sample).
            inquiry (List[str]): List describing the symbol shown in each trial.
            symbol_set (List[str]): The set of all possible symbols.

        Raises:
            SignalException: error if called before model is fit.

        Returns:
            np.array: multiplicative update term (likelihood ratios) for each symbol in the `symbol_set`.
        """
        if not self._ready_to_predict:
            raise SignalException("must use model.fit() before model.predict()")

        # Evaluate likelihood probabilities for p(e|l=1) and p(e|l=0)
        scores = np.exp(self.model.transform(data))

        # Evaluate likelihood ratios (positive class divided by negative class)
        scores = scores[:, 1] / (scores[:, 0] + 1e-10) + 1e-10

        # Apply likelihood ratios to entire symbol set.
        likelihood_ratios = np.ones(len(symbol_set))
        for idx in range(len(scores)):
            likelihood_ratios[symbol_set.index(inquiry[idx])] *= scores[idx]
        return likelihood_ratios

    def save(self, path: Path):
        """Save model weights (e.g. after training) to `path`"""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path):
        """Load pretrained model weights from `path`"""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._ready_to_predict = True
