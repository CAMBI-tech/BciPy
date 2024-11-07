"""This module defines functionality related to pre-processing simulation data.
Processed data can be subsequently sampled and provided to a SignalModel
for classification."""

import logging as logger
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Type

import numpy as np

from bcipy.acquisition import devices
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.multimodal import ContentType
from bcipy.config import (DEFAULT_DEVICE_SPEC_FILENAME,
                          DEFAULT_PARAMETERS_FILENAME, TRIGGER_FILENAME)
from bcipy.helpers.acquisition import analysis_channels, raw_data_filename
from bcipy.data.list import grouper
from bcipy.io.load import load_json_parameters, load_raw_data
from bcipy.data.parameters import Parameters
from bcipy.data.raw_data import RawData
from bcipy.data.stimuli import update_inquiry_timing
from bcipy.data.triggers import TriggerType, trigger_decoder
from bcipy.signal.model.base_model import SignalModel
from bcipy.signal.process import (ERPTransformParams, filter_inquiries,
                                  get_default_transform)
from bcipy.signal.process.transform import Composition
from bcipy.simulator.exceptions import (DeviceSpecNotFoundError,
                                        IncompatibleDeviceSpec,
                                        IncompatibleParameters)
from bcipy.task.data import EvidenceType

log = logger.getLogger(__name__)


def load_device_data(data_folder: str,
                     content_type: str) -> Tuple[RawData, DeviceSpec]:
    """Loads the data into the RawData format for the given content type.

    Parameters
    ----------
        data_folder - session data folder with the raw data, device_spec, etc.
        content_type - content type of the data to load.

    Returns
    -------
        tuple of the raw data and the associated DeviceSpec for the device used
            to record that data.
    """
    devices_by_name = devices.load(Path(data_folder,
                                        DEFAULT_DEVICE_SPEC_FILENAME),
                                   replace=True)
    specs = [
        spec for spec in devices_by_name.values()
        if spec.is_active and spec.content_type == content_type
    ]
    if not specs:
        raise DeviceSpecNotFoundError(
            f"Suitable entry not found in {data_folder}/devices.json")
    device_spec = specs[0]
    raw_data_path = Path(data_folder, raw_data_filename(device_spec))
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
    raw_data = load_raw_data(str(raw_data_path))

    if device_spec != devices.preconfigured_device(raw_data.daq_type):
        raise IncompatibleDeviceSpec(
            "DeviceSpec in devices.json does not match the raw data.")

    return raw_data, device_spec


class DecodedTriggers(NamedTuple):
    """Extracted properties after decoding the triggers.txt file and applying
    the necessary offsets and corrections."""
    targetness: List[str]  # TriggerType
    times: List[float]
    symbols: List[str]  # symbol
    corrected_times: List[float]


@dataclass()
class ExtractedExperimentData:
    """Data from an acquisition device after reshaping and filtering."""
    source_dir: str
    inquiries: np.ndarray
    trials: np.ndarray
    labels: List
    inquiry_timing: List

    decoded_triggers: DecodedTriggers
    trials_per_inquiry: int

    @property
    def trials_by_inquiry(self) -> List[np.ndarray]:
        """EEG data by inquiry.
        shape (i_inquiry, n_channel, m_trial, x_sample)
        """
        return np.split(self.trials, self.inquiries.shape[1], 1)

    @property
    def symbols_by_inquiry(self) -> List[List[str]]:
        """Get symbols by inquiry. shape (i_inquiry, s_alphabet_subset)"""
        trigger_symbols = self.decoded_triggers.symbols
        return [
            list(group) for group in grouper(
                trigger_symbols, self.trials_per_inquiry, incomplete="ignore")
        ]

    @property
    def labels_by_inquiry(self) -> List[List[int]]:
        """Shape (i_inquiry, s_alphabet_subset)"""
        return self.labels


class TimingParams(NamedTuple):
    """Timing-related parameters used in parsing the raw data into inquiries and trials."""
    # window (in seconds) of data collection after each stimulus presentation.
    trial_window: Tuple[float, float]
    # seconds before the start of the inquiry to use.
    prestim_length: float

    # delay time (in seconds) between the final stimulus in one inquiry and the
    # beginning (target stimulus or fixation cross) of the next inquiry.
    task_buffer_length: float
    stim_length: int
    time_flash: float

    @property
    def trials_per_inquiry(self) -> int:
        """Alias for stim_length"""
        return self.stim_length

    @property
    def buffer(self) -> float:
        """The task buffer length defines the min time between two inquiries
        We use half of that time here to buffer during transforms"""
        return self.task_buffer_length / 2

    @property
    def window_length(self) -> float:
        """window (in seconds) of data collection after each stimulus presentation"""
        start, end = self.trial_window
        return end - start

    def __str__(self):
        return (
            f"Timing settings: \n"
            f"Trial Window: {self.trial_window[0]} to {self.trial_window[1]}s, "
            f"Prestimulus Buffer: {self.prestim_length}s, Poststimulus Buffer: {self.buffer}s \n"
        )


def decode_triggers(
        data_folder: str,
        timing_params: TimingParams,
        device_offset: float = 0.0,
        excluded_triggers: Optional[List[TriggerType]] = None
) -> DecodedTriggers:
    """Decode the triggers.txt file in the given directory."""
    if not excluded_triggers:
        excluded_triggers = [
            TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION
        ]
    targetness, times, symbols = trigger_decoder(
        offset=device_offset,
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=excluded_triggers,
    )
    corrected = [time + timing_params.trial_window[0] for time in times]
    return DecodedTriggers(targetness=targetness,
                           times=times,
                           symbols=symbols,
                           corrected_times=corrected)


class ReshapedData():
    """Represents data components after reshaping"""

    def __init__(self):
        self.inquiries: np.ndarray = np.array([])
        self.inquiry_labels: np.ndarray = np.array([])
        self.inquiry_timing: List[List[int]] = []


class RawDataProcessor():
    """Processes raw data for a given device and converts it to
    ExtractedExperimentData for use in the simulator.

    The main API method, `process`, provides a template method that can be
    specialized by subclasses. This method performs the following for a given
    data directory.

    1. Loads the data file as a RawData object
    2. Extract timing information from trigger file based on the device of interest.
    3. Reshape and label the data.
    4. Apply filtering.
    5. Extract trials.

    Parameters
    ----------
        model - signal model which will be used to classify the resulting data
    """

    def __init__(self, model: SignalModel):
        self.check_model_compatibility(model)
        self.model = model

    @property
    def consumes(self) -> ContentType:
        """ContentType of the data that should be input"""

    @property
    def produces(self) -> EvidenceType:
        """Type of evidence that is output"""
        raise NotImplementedError

    @property
    def content_type(self):
        """Get the content_type of the data."""
        return self.consumes

    @property
    def model_device(self) -> DeviceSpec:
        """Device used to collect data for training the model."""
        return self.model.metadata.device_spec

    @property
    def reshaper(self):
        """data reshaper"""
        return self.model.reshaper

    def check_model_compatibility(self, model: SignalModel) -> None:
        """Check that the given model is compatible with this processor.
        Checked on initialization."""
        assert model.metadata, "Metadata missing from signal model."
        assert ContentType(
            model.metadata.device_spec.content_type
        ) == self.consumes, "DataProcessor is not compatible with the given model"

    def check_data_compatibility(self, data_device: DeviceSpec,
                                 sim_timing_params: TimingParams,
                                 data_timing_params: TimingParams) -> None:
        """Check the compatibility of a dataset with the model."""
        if not self.devices_compatible(self.model_device, data_device):
            raise IncompatibleDeviceSpec("Devices are not compatible")
        if not self.parameters_compatible(sim_timing_params,
                                          data_timing_params):
            raise IncompatibleParameters(
                "Timing parameters are not compatible")

        if data_device.static_offset == devices.DEFAULT_STATIC_OFFSET:
            log.warning(' '.join([
                f"Using the default static offset to decode triggers for {data_device.name}.",
                "Please make sure the correct offset is included in the devices.json file."
            ]))

    def process(self, data_folder: str,
                parameters: Parameters) -> ExtractedExperimentData:
        """Load and process the data.

        Parameters
        ----------
            data_folder - session directory for a given dataset
                (contains raw_data, devices.json, and parameters.json).
            parameters - parameters.json file for the calibration session used
                to train the model / run the simulation.
        """
        raw_data, device_spec = load_device_data(data_folder,
                                                 self.content_type.name)
        data_parameters = load_json_parameters(
            f"{data_folder}/{DEFAULT_PARAMETERS_FILENAME}", value_cast=True)

        timing_params = parameters.instantiate(TimingParams)

        self.check_data_compatibility(
            data_device=device_spec,
            sim_timing_params=timing_params,
            data_timing_params=data_parameters.instantiate(TimingParams))

        decoded_triggers = decode_triggers(data_folder, timing_params,
                                           device_spec.static_offset,
                                           self.excluded_triggers())

        reshaped_data = self.reshape_data(raw_data, decoded_triggers,
                                          timing_params)

        filtered_reshaped_data, new_sample_rate = self.apply_filters(
            reshaped_data, parameters, raw_data.sample_rate)

        trials = self.extract_trials(filtered_reshaped_data, timing_params,
                                     new_sample_rate)
        return ExtractedExperimentData(
            data_folder,
            filtered_reshaped_data.inquiries,
            trials,
            [list(arr) for arr in filtered_reshaped_data.inquiry_labels],
            filtered_reshaped_data.inquiry_timing,
            decoded_triggers=decoded_triggers,
            trials_per_inquiry=timing_params.trials_per_inquiry)

    @abstractmethod
    def reshape_data(self, raw_data: RawData,
                     decoded_triggers: DecodedTriggers,
                     timing_params: TimingParams) -> ReshapedData:
        """Use the configured reshaper to reshape the data."""
        raise NotImplementedError

    def apply_filters(self, reshaped_data: ReshapedData,
                      parameters: Parameters,
                      data_sample_rate: int) -> Tuple[ReshapedData, int]:
        """Apply any filters to the reshaped data.

        Returns
        -------
            inquiry data after filtering, updated sample_rate
        """
        raise NotImplementedError

    def extract_trials(self, filtered_reshaped_data: ReshapedData,
                       timing_params: TimingParams,
                       updated_sample_rate: float) -> np.ndarray:
        """Extract trial data from the filtered, reshaped data.

        Returns
        -------
            np.ndarray of shape (Channels, Trials, Samples)
        """
        raise NotImplementedError

    def excluded_triggers(self):
        """Trigger types to exclude when decoding"""
        return [TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION]

    def devices_compatible(self, model_device: DeviceSpec,
                           data_device: DeviceSpec) -> bool:
        """Check compatibility between the device on which the model was trained
        and the device used for data collection."""

        # TODO: check analysis channels?
        return model_device.sample_rate == data_device.sample_rate

    def parameters_compatible(self, sim_timing_params: TimingParams,
                              data_timing_params: TimingParams) -> bool:
        """Check compatibility between the parameters used for simulation and
        those used for data collection."""
        return sim_timing_params.time_flash == data_timing_params.time_flash


class EegRawDataProcessor(RawDataProcessor):
    """RawDataProcessor that processes EEG data."""
    consumes = ContentType.EEG
    produces = EvidenceType.ERP

    def reshape_data(self, raw_data: RawData,
                     decoded_triggers: DecodedTriggers,
                     timing_params: TimingParams) -> ReshapedData:
        """Use the configured reshaper to reshape the data."""

        data, _fs = raw_data.by_channel()
        channel_map = analysis_channels(raw_data.channels, self.model_device)
        inquiries, inquiry_labels, inquiry_timing = self.reshaper(
            trial_targetness_label=decoded_triggers.targetness,
            timing_info=decoded_triggers.corrected_times,
            eeg_data=data,
            sample_rate=raw_data.sample_rate,
            trials_per_inquiry=timing_params.trials_per_inquiry,
            channel_map=channel_map,
            poststimulus_length=timing_params.window_length,
            prestimulus_length=timing_params.prestim_length,
            transformation_buffer=timing_params.buffer,
        )
        reshaped = ReshapedData()
        reshaped.inquiries = inquiries
        reshaped.inquiry_labels = inquiry_labels
        reshaped.inquiry_timing = inquiry_timing
        return reshaped

    def apply_filters(self, reshaped_data: ReshapedData,
                      parameters: Parameters,
                      data_sample_rate: int) -> Tuple[ReshapedData, int]:
        """Apply any filters to the reshaped data.

        Returns
        -------
            inquiry data after filtering, updated sample_rate
        """
        transform_params = parameters.instantiate(ERPTransformParams)
        transform = self.get_transform(transform_params, data_sample_rate)
        inquiries, fs = filter_inquiries(reshaped_data.inquiries, transform,
                                         data_sample_rate)
        inquiry_timing_ints: List[List[int]] = [
            list(map(int, inq)) for inq in reshaped_data.inquiry_timing
        ]
        filtered_reshaped_data = ReshapedData()
        filtered_reshaped_data.inquiries = inquiries
        filtered_reshaped_data.inquiry_timing = update_inquiry_timing(
            inquiry_timing_ints, transform_params.down_sampling_rate)
        filtered_reshaped_data.inquiry_labels = reshaped_data.inquiry_labels
        return filtered_reshaped_data, fs

    def extract_trials(self, filtered_reshaped_data: ReshapedData,
                       timing_params: TimingParams,
                       updated_sample_rate: float) -> np.ndarray:
        """Extract trial data from the filtered, reshaped data.

        Returns
        -------
            np.ndarray of shape (Channels, Trials, Samples)
        """
        trial_duration_samples = int(timing_params.window_length *
                                     updated_sample_rate)
        return self.reshaper.extract_trials(
            filtered_reshaped_data.inquiries, trial_duration_samples,
            filtered_reshaped_data.inquiry_timing)

    def get_transform(self, transform_params: ERPTransformParams,
                      data_sample_rate: int) -> Composition:
        """"Get the transform used for filtering the data."""
        return get_default_transform(
            sample_rate_hz=data_sample_rate,
            notch_freq_hz=transform_params.notch_filter_frequency,
            bandpass_low=transform_params.filter_low,
            bandpass_high=transform_params.filter_high,
            bandpass_order=transform_params.filter_order,
            downsample_factor=transform_params.down_sampling_rate,
        )


# Module functions for matching a model to the correct processor.
def get_processor(
        data_source: ContentType,
        evidence_type: Optional[EvidenceType] = None
) -> Type[RawDataProcessor]:
    """Returns the matching processor class.

    Parameters
    ----------
        data_source - type of data that the processor should consume
        evidence_type - type of evidence that the processor should produce.
    """
    matches = [
        cls for cls in RawDataProcessor.__subclasses__()
        if cls.consumes == data_source and (
            evidence_type is None or cls.produces == evidence_type)
    ]
    if matches:
        return matches[0]
    else:
        msg = f"Data processor not found for {data_source.name}"
        if evidence_type:
            msg += f" -> {evidence_type.name}"
        raise Exception(msg)


def find_data_processor(model: SignalModel) -> Type[RawDataProcessor]:
    """Get the DataProcessor appropriate for the given model."""
    content_type = ContentType(model.metadata.device_spec.content_type)
    # Metadata may provide an EvidenceType with a model so the same data source can
    # be used to produce multiple types of evidence (ex. alpha)
    evidence_type = None
    model_output = model.metadata.evidence_type
    if model_output:
        try:
            evidence_type = EvidenceType(model_output.upper())
        except ValueError:
            logger.error(f"Unsupported evidence type: {model_output}")

    return get_processor(content_type, evidence_type)


def init_data_processor(signal_model: SignalModel) -> RawDataProcessor:
    """Find an DataProcessor that matches the given signal_model and
    initialize it."""
    processor_class = find_data_processor(signal_model)
    return processor_class(signal_model)
