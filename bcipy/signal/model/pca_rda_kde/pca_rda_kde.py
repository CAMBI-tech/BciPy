import pickle
from pathlib import Path
from typing import List

import numpy as np
from ..base_model import ModelEvaluationReport, SignalModel
from .classifier import RegularizedDiscriminantAnalysis
from .cross_validation import cost_cross_validation_auc, cross_validation
from .density_estimation import KernelDensityEstimate
from .dimensionality_reduction import ChannelWisePrincipalComponentAnalysis
from .pipeline import Pipeline
from scipy.stats import iqr


class NotFittedError(Exception):
    pass


class PcaRdaKdeModel(SignalModel):
    def __init__(self, k_folds=10):
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
        pca = ChannelWisePrincipalComponentAnalysis(var_tol=1e-5, num_ch=train_data.shape[0])
        rda = RegularizedDiscriminantAnalysis()
        model = Pipeline()
        model.add(pca)
        model.add(rda)

        # Find the optimal gamma + lambda values
        arg_cv = cross_validation(train_data, train_labels, model=model, k_folds=self.k_folds)

        # Get the AUC using those optimized gamma + lambda
        model.pipeline[1].lam = arg_cv[0]
        model.pipeline[1].gam = arg_cv[1]
        _, sc_cv, y_cv = cost_cross_validation_auc(
            model, 1, train_data, train_labels, arg_cv, k_folds=self.k_folds, split="uniform"
        )

        # After finding cross validation scores do one more round to learn the
        # final RDA model
        model.fit(train_data, train_labels)

        # Insert the density estimates to the model and train using the cross validated
        # scores to avoid over fitting. Observe that these scores are not obtained using
        # the final model
        bandwidth = 1.06 * min(np.std(sc_cv), iqr(sc_cv) / 1.34) * np.power(train_data.shape[0], -0.2)
        model.add(KernelDensityEstimate(bandwidth=bandwidth))
        model.pipeline[-1].fit(sc_cv, y_cv)

        self.model = model
        self._ready_to_predict = True
        return self

    def evaluate(self, test_data: np.array, test_labels: np.array) -> ModelEvaluationReport:
        """
        TODO - the way AUC is (and was) calculated seems weird and needs investigation/documentation
        """
        if not self._ready_to_predict:
            raise NotFittedError()

        tmp_model = Pipeline()
        tmp_model.add(self.model.pipeline[0])
        tmp_model.add(self.model.pipeline[1])

        lam_gam = (self.model.pipeline[1].lam, self.model.pipeline[1].gam)
        tmp, _, _ = cost_cross_validation_auc(
            tmp_model, 1, test_data, test_labels, lam_gam, k_folds=self.k_folds, split="uniform"
        )
        auc = -tmp
        return ModelEvaluationReport(auc)

    def predict(self, data: np.array, inquiry: List[str], symbol_set: List[str]) -> np.array:
        if not self._ready_to_predict:
            raise NotFittedError()

        # Evaluate likelihood probabilities for p(e|l=1) and p(e|l=0)
        scores = np.exp(self.model.transform(data))

        # Evaluate likelihood ratios (positive class divided by negative class)
        scores = scores[:, 1] / (scores[:, 0] + 1e-10) + 1e-10

        # Compute likelihoods for entire symbol set.
        # Letters not seen receive likelihood of 1
        # TODO - shouldn't the unseen letters have reduce
        # This maps the likelihood distribution over the symbol_set
        #   If the letter in the symbol_set does not exist in the target string,
        #       it takes 1
        likelihood_ratios = np.ones(len(symbol_set))
        for idx in range(len(scores)):
            likelihood_ratios[symbol_set.index(inquiry[idx])] *= scores[idx]
        return likelihood_ratios

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._ready_to_predict = True
