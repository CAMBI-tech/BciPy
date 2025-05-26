# mypy: disable-error-code="override"
"""Classes and functions for extracting evidence from raw device data."""
import logging
from typing import List, Optional, Type

import numpy as np

from bcipy.acquisition.multimodal import ContentType
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.stimuli import TrialReshaper
from bcipy.helpers.load import load_json_parameters
from bcipy.signal.model import SignalModel
from bcipy.signal.model.vep_signal_model import VEPSignalModel
from bcipy.task.data import EvidenceType
from bcipy.task.exceptions import MissingEvidenceEvaluator
from bcipy.signal.process import (ERPTransformParams, get_default_transform)
from bcipy.config import DEFAULT_PARAMETERS_PATH

log = logging.getLogger(__name__)


class EvidenceEvaluator:
    """Base class for a class that can evaluate raw device data using a
    signal_model. EvidenceEvaluators are responsible for performing necessary
    preprocessing steps such as filtering and reshaping.

    Parameters
    ----------
        symbol_set: set of possible symbols presented
        signal_model: model trained using a calibration session of the same user.
    """

    def __init__(self, symbol_set: List[str], signal_model: SignalModel):
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

    def __init__(self, symbol_set: List[str], signal_model: SignalModel):
        super().__init__(symbol_set, signal_model)

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
        return self.signal_model.predict(data, symbols, self.symbol_set)

class VEPEvaluator:
    """EvidenceEvaluator that extracts VEP target likelihoods from raw EEG data.
    Does NOT implement the base EvidenceEvaluator class

    Parameters
    ----------
        signal_model: vep signal model containing the CCA templates
    """
    consumes = ContentType.EEG
    produces = EvidenceType.VEP

    def __init__(self, signal_model: VEPSignalModel):

        self.device_spec = signal_model.device_spec
        assert ContentType(
            self.device_spec.content_type
        ) == self.consumes, "evaluator is not compatible with the given model"
        self.signal_model = signal_model

        self.channel_map = analysis_channels(self.device_spec.channels,
                                             self.device_spec)
        
        parameters = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True)

        transform_params = parameters.instantiate(ERPTransformParams)

        self.transform = get_default_transform(
            sample_rate_hz=self.device_spec.sample_rate,
            notch_freq_hz=transform_params.notch_filter_frequency,
            bandpass_low=transform_params.filter_low,
            bandpass_high=transform_params.filter_high,
            bandpass_order=transform_params.filter_order,
            downsample_factor=transform_params.down_sampling_rate,
        )

    def reshape(self,
                stim_time: float,
                eeg_data: np.ndarray,
                sample_rate: int,
                offset: float = 0,
                channel_map: Optional[List[int]] = None,
                poststimulus_length: float = 0.5,
                prestimulus_length: float = 0.0) -> np.ndarray:
        """Extract trial data and labels.

        Parameters
        ----------
            stim_time (float): Timestamp of stimulus event relative to start of data
            eeg_data (np.ndarray): shape (channels, samples) preprocessed EEG data
            sample_rate (int): sample rate of preprocessed EEG data
            offset (float, optional): Any calculated or hypothesized offsets in timings.
                Defaults to 0.
            channel_map (List, optional): Describes which channels to include or discard.
                Defaults to None; all channels will be used.
            poststimulus_length (float, optional): [description]. Defaults to 0.5.

        Returns
        -------
            trial_data (np.ndarray): shape (channels, trials, samples) reshaped data
            labels (np.ndarray): integer label for each trial
        """
        # Remove the rows for channels that we are not interested in
        if channel_map:
            channels_to_remove = [idx for idx, value in enumerate(channel_map) if value == 0]
            eeg_data = np.delete(eeg_data, channels_to_remove, axis=0)

        # Number of samples we are interested per trial
        poststim_samples = int(poststimulus_length * sample_rate)
        prestim_samples = int(prestimulus_length * sample_rate)

        trigger = (stim_time + offset) * sample_rate

        # print(f"{np.shape(eeg_data)}")

        return eeg_data[:, trigger - prestim_samples : trigger + poststim_samples]

    def preprocess(self, raw_data: np.ndarray, stim_time: float,
                   window_length: float) -> np.ndarray:
        """Preprocess the inquiry data.

        Parameters
        ----------
            raw_data - C x L eeg data where C is number of channels and L is the
                signal length
            stim_time - timestamps from the start of the stimulus
            window_length - The length of the time the stimulus is presented
        """
        transformed_data, transform_sample_rate = self.transform(
            raw_data, self.device_spec.sample_rate)

        # The data from DAQ is assumed to have offsets applied
        reshaped_data = self.reshape(stim_time=stim_time,
                                     eeg_data=transformed_data,
                                     sample_rate=transform_sample_rate,
                                     channel_map=self.channel_map,
                                     poststimulus_length=window_length)
        return reshaped_data

    # pylint: disable=arguments-differ
    def evaluate(self, raw_data: np.ndarray, stim_time: float,
                 window_length: float) -> np.ndarray:
        """Evaluate the evidence.

        Parameters
        ----------
            raw_data - C x L eeg data where C is number of channels and L is the
                signal length
            stim_time - The start time of the stimulus
            window_length - The length of the time the stimulus is presented
        """
        data = self.preprocess(raw_data, stim_time, window_length)
        return self.signal_model.predict(data)


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
