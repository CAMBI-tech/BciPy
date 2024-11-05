import pickle
from pathlib import Path
from typing import List

import numpy as np

from bcipy.exceptions import SignalException
from bcipy.helpers.stimuli import InquiryReshaper
from bcipy.signal.model import ModelEvaluationReport, SignalModel
from bcipy.signal.model.classifier import RegularizedDiscriminantAnalysis
from bcipy.signal.model.cross_validation import (cost_cross_validation_auc,
                                                 cross_validation)
from bcipy.signal.model.density_estimation import KernelDensityEstimate
from bcipy.signal.model.dimensionality_reduction import \
    ChannelWisePrincipalComponentAnalysis
from bcipy.signal.model.pipeline import Pipeline


class PcaRdaKdeModel(SignalModel):
    reshaper: InquiryReshaper = InquiryReshaper()
    name = "pca_rda_kde"

    def __init__(self, k_folds: int = 10, prior_type="uniform", pca_n_components=0.9):
        self.k_folds = k_folds
        self.prior_type = prior_type
        self.pca_n_components = pca_n_components
        self.optimization_elements = 1  # number of elements to optimized (RDA)
        # min and max values for the likelihood ratio output
        self.min = 1e-2
        self.max = 1e2
        self.model = None
        self.auc = None

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
        model = Pipeline(
            [
                ChannelWisePrincipalComponentAnalysis(n_components=self.pca_n_components, num_ch=train_data.shape[0]),
                RegularizedDiscriminantAnalysis(),
            ]
        )

        # Find the optimal gamma + lambda values
        arg_cv = cross_validation(train_data, train_labels, model=model, k_folds=self.k_folds)

        # Get the AUC using those optimized gamma + lambda
        rda_index = 1  # the index in the pipeline
        model.pipeline[rda_index].lam = arg_cv[0]
        model.pipeline[rda_index].gam = arg_cv[1]
        tmp, sc_cv, y_cv = cost_cross_validation_auc(
            model, rda_index, train_data, train_labels, arg_cv, k_folds=self.k_folds, split="uniform"
        )
        self.auc = -tmp
        # After finding cross validation scores do one more round to learn the
        # final RDA model
        model.fit(train_data, train_labels)

        # Insert the density estimates to the model and train using the cross validated
        # scores to avoid over fitting. Observe that these scores are not obtained using
        # the final model
        model.add(KernelDensityEstimate(scores=sc_cv))
        model.pipeline[-1].fit(sc_cv, y_cv)

        self.model = model

        if self.prior_type == "uniform":
            self.log_prior_class_1 = self.log_prior_class_0 = np.log(0.5)
        elif self.prior_type == "empirical":
            prior_class_1 = np.sum(train_labels == 1) / len(train_labels)
            self.log_prior_class_1 = np.log(prior_class_1)
            self.log_prior_class_0 = np.log(1 - prior_class_1)
        else:
            raise ValueError("prior_type must be 'empirical' or 'uniform'")

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
        if not self.ready_to_predict:
            raise SignalException("must use model.fit() before model.evaluate()")

        tmp_model = Pipeline([self.model.pipeline[0], self.model.pipeline[1]])

        lam_gam = (self.model.pipeline[1].lam, self.model.pipeline[1].gam)
        tmp, _, _ = cost_cross_validation_auc(
            tmp_model, self.optimization_elements, test_data, test_labels,
            lam_gam, k_folds=self.k_folds, split="uniform"
        )
        auc = -tmp
        return ModelEvaluationReport(auc)

    def compute_likelihood_ratio(self, data: np.array, inquiry: List[str], symbol_set: List[str]) -> np.array:
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

        if not self.ready_to_predict:
            raise SignalException("must use model.fit() before model.predict()")

        # Evaluate likelihood probabilities for p(e|l=1) and p(e|l=0)
        log_likelihoods = self.model.transform(data)
        subset_likelihood_ratios = np.exp(log_likelihoods[:, 1] - log_likelihoods[:, 0])
        # Restrict multiplicative updates to a reasonable range
        subset_likelihood_ratios = np.clip(subset_likelihood_ratios, self.min, self.max)

        # Apply likelihood ratios to entire symbol set.
        likelihood_ratios = np.ones(len(symbol_set))
        for idx in range(len(subset_likelihood_ratios)):
            likelihood_ratios[symbol_set.index(inquiry[idx])] *= subset_likelihood_ratios[idx]
        return likelihood_ratios   # used in multimodal update

    def compute_class_probabilities(self, data: np.ndarray) -> np.ndarray:
        """Converts log likelihoods from model into class probabilities.

        Returns:
            posterior (np.ndarray): shape (num_items, 2) - for each item, the model's predicted
                probability for the two labels.
        """
        if not self.ready_to_predict:
            raise SignalException("must use model.fit() before model.predict_proba()")

        # Model originally produces p(eeg | label). We want p(label | eeg):
        #
        # p(l=1 | e) = p(e | l=1) p(l=1) / p(e)
        # log(p(l=1 | e)) = log(p(e | l=1)) + log(p(l=1)) - log(p(e))
        log_scores_class_0 = self.model.transform(data)[:, 0]
        log_scores_class_1 = self.model.transform(data)[:, 1]
        log_post_0 = log_scores_class_0 + self.log_prior_class_0
        log_post_1 = log_scores_class_1 + self.log_prior_class_1
        denom = np.logaddexp(log_post_0, log_post_1)
        log_post_0 -= denom
        log_post_1 -= denom
        log_posterior = np.stack([log_post_0, log_post_1], axis=-1)
        return log_posterior
    
    def evaluate_likelihood(self, data: np.ndarray) -> np.ndarray:
        """
        Calculates log(p(e | l)) for each trial in the data.
        p(e | l=1), p(e | l=0)
        """
        if not self.ready_to_predict:
            raise SignalException("must use model.fit() before model.predict_proba()")

        log_scores_class_0 = self.model.transform(data)[:, 0]
        log_scores_class_1 = self.model.transform(data)[:, 1]
        return np.stack([log_scores_class_0, log_scores_class_1], axis=-1)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict the most likely label for each trial in the data.

        Returns:
            predictions (np.ndarray): shape (num_items,) - the predicted label for each item.
        """
        if not self.ready_to_predict:
            raise SignalException("must use model.fit() before model.predict()")

        posterior = self.compute_class_probabilities(data)
        predictions = np.argmax(posterior, axis=1)
        return predictions
    
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Converts log likelihoods from model into class probabilities.

        Returns:
            posterior (np.ndarray): shape (num_items, 2) - for each item, the model's predicted
                probability for the two labels.
        """
        if not self.ready_to_predict:
            raise SignalException("must use model.fit() before model.predict_proba()")

        # Model originally produces p(eeg | label). We want p(label | eeg):
        #
        # p(l=1 | e) = p(e | l=1) p(l=1) / p(e)
        # log(p(l=1 | e)) = log(p(e | l=1)) + log(p(l=1)) - log(p(e))
        return self.compute_class_probabilities(data)

    def save(self, path: Path) -> None:
        """Save model weights (e.g. after training) to `path`"""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path) -> SignalModel:
        """Load pretrained model from `path`"""
        with open(path, "rb") as f:
            model = pickle.load(f)

        return model
