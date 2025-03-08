"""Process raw button press data."""

from typing import List, Tuple

import numpy as np

from bcipy.acquisition import devices
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.multimodal import ContentType
from bcipy.core.list import grouper
from bcipy.core.parameters import Parameters
from bcipy.core.raw_data import RawData
from bcipy.core.triggers import TriggerType
from bcipy.display.main import PreviewParams
from bcipy.io.load import load_raw_data
from bcipy.simulator.data.data_process import (DecodedTriggers,
                                               ExtractedExperimentData,
                                               RawDataProcessor, ReshapedData,
                                               TimingParams)
from bcipy.simulator.demo.button_press_utils import (should_press_button,
                                                     simulate_raw_data,
                                                     switch_device)
from bcipy.task.data import EvidenceType


class ButtonPressDataProcessor(RawDataProcessor):
    """Data Processor for button press data.

    Note that this class does not read raw data, but instead simulates what a
    user may have done. If a target was present in an inquiry and the button press
    mode is 'press to accept', all trials in that inquiry are given data of 1.0,
    otherwise the data value is 0.0. If the mode is 'press to skip', trials within
    inquiries that do not contain the target are given 1.0 values.
    """
    consumes = ContentType.MARKERS
    produces = EvidenceType.BTN

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

        timing_params = parameters.instantiate(TimingParams)
        button_press_mode = parameters.instantiate(
            PreviewParams).button_press_mode
        decoded_triggers = self.decode_triggers(data_folder, parameters)

        inquiry_data: List[List[float]] = []
        target_labels: List[List[int]] = []

        # chunk triggers into inquiries and iterate.
        for inquiry_triggers in grouper(decoded_triggers.triggers,
                                        timing_params.trials_per_inquiry,
                                        incomplete='ignore'):
            inquiry_trial_data = []
            inquiry_target_labels = []
            # collect data for each trial in the inquiry
            for trg in inquiry_triggers:
                target_lbl = 1 if trg.type == TriggerType.TARGET else 0
                data = 1.0 if should_press_button(inquiry_triggers,
                                                  button_press_mode) else 0.0
                inquiry_trial_data.append(data)
                inquiry_target_labels.append(target_lbl)

            inquiry_data.append(inquiry_trial_data)
            target_labels.append(inquiry_target_labels)

        # shape: (channels/1 , inquiries/inquiry_count, samples/trials_per_inquiry)
        inquiry_arr = np.array([inquiry_data])
        trial_count = inquiry_arr.shape[1] * timing_params.trials_per_inquiry

        # shape: (channels/1, trials/trial_count, samples/1)
        trial_arr = inquiry_arr.reshape((1, trial_count, 1))

        return ExtractedExperimentData(
            source_dir=data_folder,
            inquiries=inquiry_arr,
            trials=trial_arr,
            labels=target_labels,
            inquiry_timing=[],  # timing data not used
            decoded_triggers=decoded_triggers,
            trials_per_inquiry=timing_params.trials_per_inquiry)

    def load_device_data(self, data_folder: str,
                         parameters: Parameters) -> Tuple[RawData, DeviceSpec]:
        """Load the device data"""
        raw_data_path = simulate_raw_data(data_folder, parameters)
        raw_data = load_raw_data(str(raw_data_path))
        device_spec = switch_device()
        return raw_data, device_spec

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
        return reshaped_data, devices.IRREGULAR_RATE

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
