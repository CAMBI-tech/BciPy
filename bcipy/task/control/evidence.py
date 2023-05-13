"""Classes and functions for extracting evidence from raw device data."""
from typing import List, NamedTuple

import numpy as np

from bcipy.acquisition.multimodal import ContentType
from bcipy.helpers.acquisition import DeviceSpec, analysis_channels
from bcipy.helpers.stimuli import TrialReshaper
from bcipy.signal.model import SignalModel
from bcipy.signal.process import get_default_transform
from bcipy.task.data import EvidenceType
from bcipy.task.exceptions import MissingEvidenceEvaluator


class EvidenceEvaluator:
    """Base class"""

    @property
    def consumes(self) -> ContentType:
        """ContentType of the data that should be input"""

    @property
    def produces(self) -> EvidenceType:
        """Type of evidence that is output"""

    def evaluate(self, *args, **kwargs):
        """Evaluate the evidence"""


# TODO: can this go in the signal module?
class FilterParams(NamedTuple):
    """Parameters used to filter signal data"""
    notch_filter_frequency: int = 60
    filter_low: int = 2
    filter_high: int = 45
    filter_order: int = 2
    downsample_rate: int = 2


class EegEvaluator(EvidenceEvaluator):
    """Extracts symbol likelihoods from raw EEG data"""
    consumes = ContentType.EEG
    produces = EvidenceType.ERP

    def __init__(self, symbol_set: List[str], signal_model: SignalModel,
                 device_spec: DeviceSpec, filter_params: FilterParams):
        assert ContentType(
            device_spec.content_type
        ) == self.consumes, "evaluator is not compatible with the given device"
        self.symbol_set = symbol_set
        self.signal_model = signal_model
        self.filter_params = filter_params

        self.sample_rate = device_spec.sample_rate
        self.channel_map = analysis_channels(device_spec.channels, device_spec)

    # pylint: disable=arguments-differ
    def evaluate(self, raw_data: np.array, symbols: List[str],
                 times: List[float], target_info: List[str],
                 window_length: float) -> np.array:
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

        transform = get_default_transform(
            sample_rate_hz=self.sample_rate,
            notch_freq_hz=self.filter_params.notch_filter_frequency,
            bandpass_low=self.filter_params.filter_low,
            bandpass_high=self.filter_params.filter_high,
            bandpass_order=self.filter_params.filter_order,
            downsample_factor=self.filter_params.downsample_rate,
        )

        transformed_data, transform_sample_rate = transform(
            raw_data, self.sample_rate)

        # The data from DAQ is assumed to have offsets applied
        reshape = TrialReshaper()
        reshaped_data, _lbls = reshape(trial_targetness_label=target_info,
                                       timing_info=times,
                                       eeg_data=transformed_data,
                                       sample_rate=transform_sample_rate,
                                       channel_map=self.channel_map,
                                       poststimulus_length=window_length)

        return self.signal_model.predict(reshaped_data, symbols,
                                         self.symbol_set)


def get_evaluator(data_source: ContentType,
                  evidence_type: EvidenceType) -> EvidenceEvaluator:
    """Returns the matching evaluator.

    Parameters
    ----------
        data_source - type of data that the evaluator should consume
        evidence_type - type of evidence that the evaluator should produce.
    """
    matches = [
        cls for cls in EvidenceEvaluator.__subclasses__()
        if cls.consumes == data_source and cls.produces == evidence_type
    ]
    if matches:
        return matches[0]
    else:
        raise MissingEvidenceEvaluator(
            f"Evidence Evaluator not found for {data_source.name} -> {evidence_type.name}"
        )
