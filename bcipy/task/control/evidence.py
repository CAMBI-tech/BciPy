# mypy: disable-error-code="override"
"""Classes and functions for extracting evidence from raw device data."""
import logging
from typing import List, Optional, Type

import numpy as np

from bcipy.acquisition.multimodal import ContentType
from bcipy.config import SESSION_LOG_FILENAME
from bcipy.core.parameters import Parameters
from bcipy.core.stimuli import TrialReshaper
from bcipy.display.main import ButtonPressMode
from bcipy.helpers.acquisition import analysis_channels
from bcipy.signal.model import SignalModel
from bcipy.task.data import EvidenceType
from bcipy.task.exceptions import MissingEvidenceEvaluator

log = logging.getLogger(SESSION_LOG_FILENAME)


class EvidenceEvaluator:
    """Base class for a class that can evaluate raw device data using a
    signal_model. EvidenceEvaluators are responsible for performing necessary
    preprocessing steps such as filtering and reshaping.

    Parameters
    ----------
        symbol_set: set of possible symbols presented
        signal_model: model trained using a calibration session of the same user.
    """

    def __init__(self,
                 symbol_set: List[str],
                 signal_model: SignalModel,
                 parameters: Optional[Parameters] = None):
        assert signal_model.metadata, "Metadata missing from signal model."
        device_spec = signal_model.metadata.device_spec
        assert ContentType(
            signal_model.metadata.device_spec.content_type
        ) == self.consumes, "evaluator is not compatible with the given model"
        self.symbol_set = symbol_set
        self.signal_model = signal_model
        self.device_spec = device_spec

    @property
    def consumes(self) -> ContentType:
        """ContentType of the data that should be input"""

    @property
    def produces(self) -> EvidenceType:
        """Type of evidence that is output"""

    def evaluate(self, **kwargs):
        """Evaluate the evidence"""


class EEGEvaluator(EvidenceEvaluator):
    """EvidenceEvaluator that extracts symbol likelihoods from raw EEG data.

    Parameters
    ----------
        symbol_set: set of possible symbols presented
        signal_model: trained signal model
    """
    consumes = ContentType.EEG
    produces = EvidenceType.ERP

    def __init__(self,
                 symbol_set: List[str],
                 signal_model: SignalModel,
                 parameters: Optional[Parameters] = None):
        super().__init__(symbol_set, signal_model, parameters)

        self.channel_map = analysis_channels(self.device_spec.channels,
                                             self.device_spec)
        self.transform = signal_model.metadata.transform
        self.reshape = TrialReshaper()

    def preprocess(self, raw_data: np.ndarray, times: List[float],
                   target_info: List[str], window_length: float) -> np.ndarray:
        """Preprocess the inquiry data.

        Parameters
        ----------
            raw_data - C x L eeg data where C is number of channels and L is the
                signal length
            symbols - symbols displayed in the inquiry
            times - timestamps associated with each symbol
            target_info - target information about the stimuli;
                ex. ['nontarget', 'nontarget', ...]
            window_length - The length of the time between stimuli presentation
        """
        transformed_data, transform_sample_rate = self.transform(
            raw_data, self.device_spec.sample_rate)

        # The data from DAQ is assumed to have offsets applied
        reshaped_data, _lbls = self.reshape(trial_targetness_label=target_info,
                                            timing_info=times,
                                            eeg_data=transformed_data,
                                            sample_rate=transform_sample_rate,
                                            channel_map=self.channel_map,
                                            poststimulus_length=window_length)
        return reshaped_data

    # pylint: disable=arguments-differ
    def evaluate(self, raw_data: np.ndarray, symbols: List[str],
                 times: List[float], target_info: List[str],
                 window_length: float) -> np.ndarray:
        """Evaluate the evidence.

        Parameters
        ----------
            raw_data - C x L eeg data where C is number of channels and L is the
                signal length
            symbols - symbols displayed in the inquiry
            times - timestamps associated with each symbol
            target_info - target information about the stimuli;
                ex. ['nontarget', 'nontarget', ...]
            window_length - The length of the time between stimuli presentation
        """
        data = self.preprocess(raw_data, times, target_info, window_length)
        return self.signal_model.compute_likelihood_ratio(
            data, symbols, self.symbol_set)


class GazeEvaluator(EvidenceEvaluator):
    """EvidenceEvaluator that extracts symbol likelihoods from raw gaze data.

    Parameters
    ----------
        symbol_set: set of possible symbols presented
        gaze_model: trained gaze model
    """
    consumes = ContentType.EYETRACKER
    produces = EvidenceType.EYE

    def __init__(self,
                 symbol_set: List[str],
                 signal_model: SignalModel,
                 parameters: Optional[Parameters] = None):
        super().__init__(symbol_set, signal_model, parameters)

        self.channel_map = analysis_channels(self.device_spec.channels,
                                             self.device_spec)
        self.transform = signal_model.metadata.transform
        self.reshape = TrialReshaper()

    def preprocess(self, raw_data: np.ndarray, times: List[float],
                   target_info: List[str], window_length: float) -> np.ndarray:
        """Preprocess the inquiry data.

        Parameters
        ----------
            raw_data - C x L eeg data where C is number of channels and L is the
                signal length
            symbols - symbols displayed in the inquiry
            times - timestamps associated with each symbol
            target_info - target information about the stimuli;
                ex. ['nontarget', 'nontarget', ...]
            window_length - The length of the time between stimuli presentation
        """
        transformed_data, transform_sample_rate = self.transform(
            raw_data, self.device_spec.sample_rate)

        # The data from DAQ is assumed to have offsets applied
        reshaped_data, _lbls = self.reshape(trial_targetness_label=target_info,
                                            timing_info=times,
                                            eeg_data=transformed_data,
                                            sample_rate=transform_sample_rate,
                                            channel_map=self.channel_map,
                                            poststimulus_length=window_length)
        return reshaped_data

    # pylint: disable=arguments-differ
    def evaluate(self, raw_data: np.ndarray, symbols: List[str],
                 times: List[float], target_info: List[str],
                 window_length: float) -> np.ndarray:
        """Evaluate the evidence.

        Parameters
        ----------
            raw_data - C x L eeg data where C is number of channels and L is the
                signal length
            symbols - symbols displayed in the inquiry
            times - timestamps associated with each symbol
            target_info - target information about the stimuli;
                ex. ['nontarget', 'nontarget', ...]
            window_length - The length of the time between stimuli presentation
        """
        data = self.preprocess(raw_data, times, target_info, window_length)
        # We need the likelihoods in the form of p(label | gaze). predict returns the argmax of the likelihoods.
        # Therefore we need predict_proba method to get the likelihoods.
        return self.signal_model.evaluate_likelihood(
            data)  # multiplication over the inquiry


class SwitchEvaluator(EvidenceEvaluator):
    """EvidenceEvaluator that extracts symbol likelihoods from raw Switch data.

    Parameters
    ----------
        symbol_set: set of possible symbols presented
        signal_model: trained signal model
    """
    consumes = ContentType.MARKERS
    produces = EvidenceType.BTN

    def __init__(self,
                 symbol_set: List[str],
                 signal_model: SignalModel,
                 parameters: Optional[Parameters] = None):
        super().__init__(symbol_set, signal_model, parameters)
        self.button_press_mode = ButtonPressMode(
            parameters.get('preview_inquiry_progress_method'))
        self.trial_count = parameters.get('stim_length')

    def preprocess(self, raw_data: np.ndarray, times: List[float],
                   target_info: List[str], window_length: float) -> np.ndarray:
        """Preprocess the inquiry data.

        Determines the return data based on whether the switch was pressed
        during the inquiry and the configured ButtonPressMode.
        """
        switch_was_pressed = np.any(raw_data)

        # shape: (channels/1, trials/trial_count, samples/1)
        data_shape = (1, self.trial_count, 1)
        ones = np.ones(data_shape)
        zeros = np.zeros(data_shape)

        # Inquiries with 1.0s will be upgraded/supported by the model probabilities.
        rules = {
            ButtonPressMode.NOTHING:
            lambda: ones,
            ButtonPressMode.ACCEPT:
            lambda: ones if switch_was_pressed else zeros,
            ButtonPressMode.REJECT:
            lambda: ones if not switch_was_pressed else zeros
        }
        return rules[self.button_press_mode]()

    # pylint: disable=arguments-differ
    def evaluate(self, raw_data: np.ndarray, symbols: List[str],
                 times: List[float], target_info: List[str],
                 window_length: float) -> np.ndarray:
        """Evaluate the evidence.

        Parameters
        ----------
            raw_data - C x L eeg data where C is number of channels and L is the
                signal length
            symbols - symbols displayed in the inquiry
            times - timestamps associated with each symbol
            target_info - target information about the stimuli;
                ex. ['nontarget', 'nontarget', ...]
            window_length - The length of the time between stimuli presentation
        """
        data = self.preprocess(raw_data, times, target_info, window_length)
        return self.signal_model.compute_likelihood_ratio(
            data, symbols, self.symbol_set)


def get_evaluator(
        data_source: ContentType,
        evidence_type: Optional[EvidenceType] = None
) -> Type[EvidenceEvaluator]:
    """Returns the matching evaluator class.

    Parameters
    ----------
        data_source - type of data that the evaluator should consume
        evidence_type - type of evidence that the evaluator should produce.
    """
    matches = [
        cls for cls in EvidenceEvaluator.__subclasses__()
        if cls.consumes == data_source and (
            evidence_type is None or cls.produces == evidence_type)
    ]
    if matches:
        return matches[0]
    else:
        msg = f"Evidence Evaluator not found for {data_source.name}"
        if evidence_type:
            msg += f" -> {evidence_type.name}"
        raise MissingEvidenceEvaluator(msg)


def find_matching_evaluator(
        signal_model: SignalModel) -> Type[EvidenceEvaluator]:
    """Find the first EvidenceEvaluator compatible with the given signal
    model."""
    content_type = ContentType(signal_model.metadata.device_spec.content_type)
    # Metadata may provide an EvidenceType with a model so the same data source can
    # be used to produce multiple types of evidence (ex. alpha)
    evidence_type = None
    model_output = signal_model.metadata.evidence_type
    if model_output:
        try:
            evidence_type = EvidenceType(model_output.upper())
        except ValueError:
            log.error(f"Unsupported evidence type: {model_output}")

    return get_evaluator(content_type, evidence_type)


def init_evidence_evaluator(symbol_set: List[str],
                            signal_model: SignalModel) -> EvidenceEvaluator:
    """Find an EvidenceEvaluator that matches the given signal_model and
    initialize it."""
    evaluator_class = find_matching_evaluator(signal_model)
    return evaluator_class(symbol_set, signal_model)
