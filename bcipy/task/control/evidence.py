"""Evidence evaluation module for BCI task control.

This module provides classes and functions for extracting evidence from raw device data,
including EEG, gaze tracking, and switch input data. The module supports different types
of evidence evaluation based on the input data type and desired output evidence type.
"""

# mypy: disable-error-code="override"
import logging
from typing import List, Optional, Type, Any

import numpy as np

from bcipy.acquisition.multimodal import ContentType
from bcipy.config import SESSION_LOG_FILENAME
from bcipy.core.parameters import Parameters
from bcipy.core.stimuli import GazeReshaper, TrialReshaper
from bcipy.display.main import ButtonPressMode
from bcipy.helpers.acquisition import analysis_channels
from bcipy.signal.model import SignalModel
from bcipy.signal.process import extract_eye_info
from bcipy.task.data import EvidenceType
from bcipy.task.exceptions import MissingEvidenceEvaluator

log = logging.getLogger(SESSION_LOG_FILENAME)


class EvidenceEvaluator:
    """Base class for evaluating raw device data using a signal model.

    This class defines the interface for evidence evaluators, which are responsible
    for performing necessary preprocessing steps such as filtering and reshaping
    before evaluating the evidence.

    Attributes:
        symbol_set: List of possible symbols that can be presented.
        signal_model: Model trained using calibration session data.
        device_spec: Specification of the input device.
    """

    def __init__(
            self,
            symbol_set: List[str],
            signal_model: SignalModel,
            parameters: Optional[Parameters] = None) -> None:
        """Initialize the evidence evaluator.

        Args:
            symbol_set: List of possible symbols that can be presented.
            signal_model: Model trained using calibration session data.
            parameters: Optional configuration parameters.

        Raises:
            AssertionError: If signal model metadata is missing or incompatible.
        """
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
        """Get the type of data this evaluator consumes.

        Returns:
            ContentType: Type of input data required.
        """
        raise NotImplementedError()

    @property
    def produces(self) -> EvidenceType:
        """Get the type of evidence this evaluator produces.

        Returns:
            EvidenceType: Type of evidence output.
        """
        raise NotImplementedError()

    def evaluate(self, **kwargs: Any) -> np.ndarray:
        """Evaluate the evidence from raw data.

        Args:
            **kwargs: Arbitrary keyword arguments for evaluation.

        Returns:
            np.ndarray: Evaluated evidence data.
        """
        raise NotImplementedError()


class EEGEvaluator(EvidenceEvaluator):
    """Evidence evaluator for extracting symbol likelihoods from EEG data.

    This evaluator processes raw EEG data to compute likelihood ratios for
    different symbols based on the ERP response.

    Attributes:
        consumes: Type of input data (EEG).
        produces: Type of evidence output (ERP).
        channel_map: Mapping of EEG channels.
        transform: Signal transformation function.
        reshape: Trial reshaping function.
    """

    consumes = ContentType.EEG
    produces = EvidenceType.ERP

    def __init__(
            self,
            symbol_set: List[str],
            signal_model: SignalModel,
            parameters: Optional[Parameters] = None) -> None:
        """Initialize the EEG evaluator.

        Args:
            symbol_set: List of possible symbols that can be presented.
            signal_model: Model trained using calibration session data.
            parameters: Optional configuration parameters.
        """
        super().__init__(symbol_set, signal_model, parameters)

        self.channel_map = analysis_channels(self.device_spec.channels,
                                             self.device_spec)
        self.transform = signal_model.metadata.transform
        self.reshape = TrialReshaper()

    def preprocess(
            self,
            raw_data: np.ndarray,
            times: List[float],
            target_info: List[str],
            window_length: float) -> np.ndarray:
        """Preprocess the inquiry EEG data.

        Args:
            raw_data: C x L EEG data where C is number of channels and L is
                signal length.
            times: Timestamps associated with each symbol.
            target_info: Target information about the stimuli
                (e.g. ['nontarget', 'nontarget', ...]).
            window_length: Length of time between stimuli presentation.

        Returns:
            np.ndarray: Preprocessed EEG data.
        """
        transformed_data, transform_sample_rate = self.transform(
            raw_data, self.device_spec.sample_rate)

        # The data from DAQ is assumed to have offsets applied
        reshaped_data, _lbls = self.reshape(
            trial_targetness_label=target_info,
            timing_info=times,
            eeg_data=transformed_data,
            sample_rate=transform_sample_rate,
            channel_map=self.channel_map,
            poststimulus_length=window_length)
        return reshaped_data

    def evaluate(
            self,
            raw_data: np.ndarray,
            symbols: List[str],
            times: List[float],
            target_info: List[str],
            window_length: float,
            *args: Any) -> np.ndarray:
        """Evaluate EEG evidence.

        Args:
            raw_data: C x L EEG data where C is number of channels and L is
                signal length.
            symbols: Symbols displayed in the inquiry.
            times: Timestamps associated with each symbol.
            target_info: Target information about the stimuli
                (e.g. ['nontarget', 'nontarget', ...]).
            window_length: Length of time between stimuli presentation.
            *args: Additional arguments.

        Returns:
            np.ndarray: Likelihood ratios for each symbol.
        """
        data = self.preprocess(raw_data, times, target_info, window_length)
        return self.signal_model.compute_likelihood_ratio(
            data, symbols, self.symbol_set)


class GazeEvaluator(EvidenceEvaluator):
    """Evidence evaluator for extracting symbol likelihoods from gaze data.

    This evaluator processes raw eye tracking data to compute likelihoods
    for different symbols based on gaze patterns.

    Attributes:
        consumes: Type of input data (EYETRACKER).
        produces: Type of evidence output (EYE).
        channel_map: Mapping of eye tracking channels.
        transform: Signal transformation function.
        reshape: Gaze data reshaping function.
    """

    consumes = ContentType.EYETRACKER
    produces = EvidenceType.EYE

    def __init__(
            self,
            symbol_set: List[str],
            signal_model: SignalModel,
            parameters: Optional[Parameters] = None) -> None:
        """Initialize the gaze evaluator.

        Args:
            symbol_set: List of possible symbols that can be presented.
            signal_model: Model trained using calibration session data.
            parameters: Optional configuration parameters.
        """
        super().__init__(symbol_set, signal_model, parameters)

        self.channel_map = analysis_channels(self.device_spec.channels,
                                             self.device_spec)
        self.transform = signal_model.metadata.transform
        self.reshape = GazeReshaper()

    def preprocess(
            self,
            raw_data: np.ndarray,
            times: List[float],
            flash_time: float) -> np.ndarray:
        """Preprocess the inquiry gaze data.

        The preprocessing is functionally different than Gaze Reshaper, since
        the raw data contains only one inquiry. start_idx is determined as the
        start time of first symbol flashing multiplied by the sampling rate
        of eye tracker. stop_idx is the index indicating the end of last
        symbol flashing.

        Args:
            raw_data: C x L data where C is number of channels and L is signal
                length. Includes all channels in devices.json.
            times: Timestamps associated with each symbol.
            flash_time: Duration (in seconds) of each stimulus.

        Returns:
            np.ndarray: Preprocessed gaze data (4, N_samples).
        """
        if self.transform:
            transformed_data, transform_sample_rate = self.transform(
                raw_data, self.device_spec.sample_rate)
        else:
            transformed_data = raw_data
            transform_sample_rate = self.device_spec.sample_rate

        start_idx = int(self.device_spec.sample_rate * times[0])
        stop_idx = start_idx + int(
            (times[-1] - times[0] + flash_time) * self.device_spec.sample_rate)
        data_all_channels = transformed_data[:, start_idx:stop_idx]

        # Extract left and right eye from all channels. Remove/replace nan values
        left_eye, right_eye, _, _, _, _ = extract_eye_info(data_all_channels)
        reshaped_data = np.vstack((np.array(left_eye).T, np.array(right_eye).T))

        return reshaped_data

    def evaluate(
            self,
            raw_data: np.ndarray,
            symbols: List[str],
            times: List[float],
            target_info: List[str],
            window_length: float,
            flash_time: float,
            stim_length: float) -> np.ndarray:
        """Evaluate gaze evidence.

        Args:
            raw_data: C x L data where C is number of channels and L is signal
                length.
            symbols: Symbols displayed in the inquiry.
            times: Timestamps associated with each symbol.
            target_info: Target information about the stimuli.
            window_length: Length of time between stimuli presentation.
            flash_time: Duration of each stimulus.
            stim_length: Length of stimulus sequence.

        Returns:
            np.ndarray: Likelihood values for each symbol.
        """
        data = self.preprocess(raw_data, times, flash_time)
        # We need the likelihoods in the form of p(label | gaze).
        # predict returns the argmax of the likelihoods.
        # Therefore we need predict_proba method to get the likelihoods.
        likelihood = self.signal_model.evaluate_likelihood(
            data, symbols, self.symbol_set)
        return likelihood


class SwitchEvaluator(EvidenceEvaluator):
    """Evidence evaluator for extracting symbol likelihoods from switch data.

    This evaluator processes raw switch input data to compute likelihoods
    for different symbols based on button press patterns.

    Attributes:
        consumes: Type of input data (MARKERS).
        produces: Type of evidence output (BTN).
        button_press_mode: Mode of button press interpretation.
        trial_count: Number of trials in stimulus sequence.
    """

    consumes = ContentType.MARKERS
    produces = EvidenceType.BTN

    def __init__(
            self,
            symbol_set: List[str],
            signal_model: SignalModel,
            parameters: Optional[Parameters] = None) -> None:
        """Initialize the switch evaluator.

        Args:
            symbol_set: List of possible symbols that can be presented.
            signal_model: Model trained using calibration session data.
            parameters: Optional configuration parameters.

        Raises:
            AssertionError: If button press mode is not supported.
        """
        super().__init__(symbol_set, signal_model, parameters)
        if not parameters:
            raise ValueError("Parameters required for SwitchEvaluator")

        self.button_press_mode = ButtonPressMode(
            parameters.get('preview_inquiry_progress_method'))
        self.trial_count = parameters.get('stim_length')

        if self.button_press_mode == ButtonPressMode.NOTHING:
            raise AssertionError((
                "Button press mode not supported.",
                "To run without button press evidence set the acq_mode to exclude MARKERS."
            ))

    def preprocess(
            self,
            raw_data: np.ndarray,
            times: List[float],
            target_info: List[str],
            window_length: float) -> np.ndarray:
        """Preprocess the inquiry switch data.

        Determines the return data based on whether the switch was pressed
        during the inquiry and the configured ButtonPressMode.

        Args:
            raw_data: Switch input data.
            times: Timestamps associated with each symbol.
            target_info: Target information about the stimuli.
            window_length: Length of time between stimuli presentation.

        Returns:
            np.ndarray: Preprocessed switch data.
        """
        switch_was_pressed = np.any(raw_data)

        # shape: (channels/1, trials/trial_count, samples/1)
        data_shape = (1, self.trial_count, 1)
        ones = np.ones(data_shape)
        zeros = np.zeros(data_shape)

        # Inquiries with 1.0s will be upgraded/supported by the model probabilities.
        rules = {
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
        probs = self.signal_model.compute_likelihood_ratio(
            data, symbols, self.symbol_set)
        return probs


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


def init_evidence_evaluator(
        symbol_set: List[str],
        signal_model: SignalModel,
        parameters: Optional[Parameters] = None) -> EvidenceEvaluator:
    """Find an EvidenceEvaluator that matches the given signal_model and
    initialize it."""
    evaluator_class = find_matching_evaluator(signal_model)
    return evaluator_class(symbol_set, signal_model, parameters)
