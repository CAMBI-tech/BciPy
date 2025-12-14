"""Process raw button press data."""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from bcipy.acquisition import devices
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.multimodal import ContentType
from bcipy.config import TRIGGER_FILENAME
from bcipy.core.list import grouper
from bcipy.core.parameters import Parameters
from bcipy.core.raw_data import RawData
from bcipy.core.triggers import TriggerType
from bcipy.display.main import ButtonPressMode, PreviewParams
from bcipy.helpers.acquisition import raw_data_filename
from bcipy.io.load import load_raw_data
from bcipy.simulator.data.data_process import (DecodedTriggers,
                                               ExtractedExperimentData,
                                               RawDataProcessor, ReshapedData,
                                               TimingParams)
from bcipy.simulator.exceptions import IncompatibleData, IncompatibleParameters
from bcipy.simulator.util.switch_utils import inquiry_windows, switch_device
from bcipy.task.data import EvidenceType


class SwitchDataProcessor(RawDataProcessor):
    """Data Processor for button press data.

    Note that this class reads raw switch data. If this is not present in the
    data_folder it can be generated from Inquiry Preview triggers or simulated.
    See the switch_utils module for details.

    The processed data values associated with a switch press depend on the button
    press mode configured parameters.

    If the mode is 'press to accept/confirm' and the switch is pressed, all trials
    in that inquiry are given a data value of 1.0. Otherwise, the data value is 0.0.

    If the mode is 'press to skip', all trials in inquiries with a button press are
    given a value 0.0. Otherwise, the data value is 1.0.

    The switch model will upvote all symbols in an inquiry if any of the symbols have
    a value of 1.0, and will downvote if all are 0.0.

    This also means that this processor should only be used in conjunction with the
    InquirySampler.
    """
    consumes = ContentType.MARKERS
    produces = EvidenceType.BTN

    def data_value(self, raw_data: RawData, inq_start: float, inq_stop: float,
                   button_press_mode: ButtonPressMode) -> float:
        """Returns the data value for all trials in the inquiry.

        The switch was pressed some time during the inquiry but not necessarily
        coinciding with a specific symbol, so all symbols are assigned the same
        value (1.0 or 0.0).

        The value depends on the button press mode and indicates whether the
        symbols in the inquiry should be upgraded or downgraded.
        """
        times = raw_data.dataframe['lsl_timestamp']
        switch_was_pressed = any(inq_start <= time <= inq_stop
                                 for time in times)

        rules = {
            ButtonPressMode.NOTHING: lambda: 1.0,
            ButtonPressMode.ACCEPT: lambda: 1.0 if switch_was_pressed else 0.0,
            ButtonPressMode.REJECT: lambda: 1.0
            if not switch_was_pressed else 0.0
        }
        return rules[button_press_mode]()

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
        raw_data, _spec = self.load_device_data(data_folder, parameters)

        timing_params = parameters.instantiate(TimingParams)
        button_press_mode = parameters.instantiate(
            PreviewParams).button_press_mode

        if button_press_mode is ButtonPressMode.NOTHING:
            raise IncompatibleParameters("Button press mode must be set.")

        decoded_triggers = self.decode_triggers(data_folder, parameters)
        time_ranges = inquiry_windows(Path(data_folder, TRIGGER_FILENAME),
                                      timing_params.time_flash)

        inquiry_data: List[List[float]] = []
        target_labels: List[List[int]] = []

        # chunk triggers into inquiries and iterate.
        for i, inquiry_triggers in enumerate(
                grouper(decoded_triggers.triggers,
                        timing_params.trials_per_inquiry,
                        incomplete='ignore')):
            inquiry_trial_data = []
            inquiry_target_labels = []

            # Returns the same value for all trials in the inquiry.
            start, stop = time_ranges[i]
            data = self.data_value(raw_data, start, stop, button_press_mode)

            # collect data for each trial in the inquiry
            for trg in inquiry_triggers:
                target_lbl = 1 if trg.type == TriggerType.TARGET else 0
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
        device_spec = switch_device()
        raw_data_path = Path(data_folder, raw_data_filename(device_spec))
        if not raw_data_path.exists():
            raise IncompatibleData(
                "Missing raw data for switch. Use switch_utils to generate.")
        raw_data = load_raw_data(str(raw_data_path))
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
