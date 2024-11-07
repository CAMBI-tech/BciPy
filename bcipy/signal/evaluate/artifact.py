# mypy: disable-error-code="assignment,arg-type"
import os
from enum import Enum
from pathlib import Path
from typing import Union, List, Tuple, Optional
from logging import getLogger
import logging

from bcipy.config import (
    DEFAULT_PARAMETERS_FILENAME,
    RAW_DATA_FILENAME,
    TRIGGER_FILENAME,
    DEFAULT_DEVICE_SPEC_FILENAME,
    BCIPY_ROOT,
    SESSION_LOG_FILENAME
)
from bcipy.helpers.acquisition import analysis_channels
from bcipy.io.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from bcipy.data.stimuli import mne_epochs
from bcipy.io.convert import convert_to_mne
from bcipy.data.raw_data import RawData
from bcipy.signal.process import get_default_transform
from bcipy.data.triggers import TriggerType, trigger_decoder
import bcipy.acquisition.devices as devices
from bcipy.acquisition.devices import DeviceSpec

import mne
mne.set_log_level('WARNING')
log = getLogger(SESSION_LOG_FILENAME)

from mne import Annotations


class DefaultArtifactParameters(Enum):
    """Default Artifact Parameters.

    These values are used as defaults for artifact analysis purposes. These values were determined through
    experimentation and are used as a starting point for artifact detection. These values should be adjusted
    based on the data being analyzed.
    """

    # Voltage
    PEAK_THRESHOLD = 15e-6
    PEAK_MIN_DURATION = 0.001
    FLAT_THRESHOLD = 0.5e-6
    FLAT_MIN_DURATION = 0.1

    VOlTAGE_LABEL_DURATION = 0.25

    # Eye
    EOG_THRESHOLD = 55e-6
    EOG_MIN_DURATION = 0.5

    EOG_LABEL_DURATION = 0.25

    # I/O
    ARTIFACT_LABELLED_FILENAME = 'artifacts.fif'

    def __str__(self) -> str:
        """Return the value of the artifact type."""
        return self.value

    def __repr__(self) -> str:
        """Return the class name and the artifact type name."""
        return f'{self.__class__.__name__}.{self.name}'


class ArtifactType(Enum):
    """Artifact Type Enum.

    The artifact type enum is used to define the types of artifacts that can be detected in the data.
    """

    VOLTAGE = 'voltage'
    EOG = 'eog'
    BLINK = 'blink'
    ECG = 'ecg'
    EMG = 'emg'
    FLAT = 'flat'
    EVENT = 'event'

    def __str__(self) -> str:
        """Return the value of the artifact type."""
        return self.value

    def __repr__(self) -> str:
        """Return the class name and the artifact type name."""
        return f'{self.__class__.__name__}.{self.name}'

    def label(self) -> str:
        """Label the artifact type.

        Returns
        -------
        label - string - The label for the artifact type with the prefix BAD.
                         This is used for epoch handling by MNE.
        """
        return f'BAD_{self.value}'


class ArtifactDetection:
    """Artifact Detection Class.

    The artifact detection class is used to detect and label artifacts in the raw data. This class is designed to be
    used in a pipeline for artifact detection and labelling.


    Parameters
    ----------
    raw_data : RawData
        The raw data to process for artifacts.

    parameters : dict
        The parameters used in the collection of the raw data.

    device_spec : dict
        The device specification for the raw data.

    save_path : optional str
        The path to save the artifact labelled data. If none provided, data is not saved.

    session_triggers : optional list
        Optional list of session triggers to use for co-labelling the data. This is useful for determining the impact
        of artifacts on the relevant epochs.

    percent_bad : float
        The percentage of bad epochs to allow before channel exclusion. Defaults to 50%.

    detect_eog : bool
        Whether to detect EOG artifacts. Defaults to True. Eye channels must be provided if this is set to True.

    eye_channels : list
        The list of eye channels to use for detecting EOG artifacts. Defaults to None. If provided, the channels should
        look like ['Fp1', 'Fp2', 'F7', 'F8'].

    detect_voltage : bool
        Whether to detect voltage artifacts. Defaults to True.

    semi_automatic : bool
        Whether to use a semi-automatic approach to artifact detection. Defaults to False.

    session_triggers : tuple
        A tuple of lists containing the trigger type, trigger timing, and trigger label for the session.
    """

    supported_units: List[str] = ['volts', 'microvolts']
    support_device_types: List[str] = ['EEG']
    analysis_done: bool = False
    dropped: Optional[str] = None
    eog_annotations: Optional[Annotations] = None
    voltage_annotations: Optional[Annotations] = None

    def __init__(
            self,
            raw_data: RawData,
            parameters: dict,
            device_spec: DeviceSpec,
            save_path: Optional[str] = None,
            semi_automatic: bool = False,
            session_triggers: Optional[Tuple[list, list, list]] = None,
            percent_bad: float = 50.0,
            detect_eog: bool = True,
            eye_channels: Optional[List[str]] = None,
            detect_voltage: bool = True) -> None:
        self.raw_data = raw_data
        self.total_time = raw_data.total_seconds
        self.device_spec = device_spec
        assert self.device_spec.content_type in self.support_device_types, \
            f'BciPy Artifact Analysis only supports {self.support_device_types} data at this time.'
        self.eye_channels = eye_channels
        self.percent_bad = percent_bad
        self.parameters = parameters

        # get the trial window from the parameters, in older versions it was not included default to 0.5
        start_trial, end_trial = self.parameters.get('trial_window', (0, 0.5))
        self.trial_duration = end_trial - start_trial

        self.session_triggers = None
        if session_triggers:
            self.triggers = session_triggers
            self.trigger_description = session_triggers[0]
            self.trigger_time = session_triggers[1]
            self.trigger_label = session_triggers[2]

            self.session_triggers = mne.Annotations(
                self.trigger_time, [self.trial_duration] * len(self.trigger_time), self.trigger_description)

        assert len(device_spec.channel_specs) > 0, 'DeviceSpec used must have channels. None found.'
        self.units = device_spec.channel_specs[0].units
        log.info(f'Artifact detection using {self.units} units.')
        assert self.units in self.supported_units, \
            f'Data loaded in units that cannot be processed. Support units={self.supported_units}'
        # MNE assumes that data is recorded in Volts. However, many devices use microvolts.
        # If this value is False, a conversion to volts will be required.
        self.volts = True if self.units == 'volts' else False
        self.mne_data = self.raw_data_to_mne(raw_data, volts=self.volts)
        self.detect_voltage = detect_voltage
        self.detect_eog = detect_eog
        self.semi_automatic = semi_automatic

        log.info(f'Artifact detection with {self.detect_voltage=}, {self.detect_eog=}, {self.semi_automatic=}')

        self.save_path = save_path

    def detect_artifacts(self) -> Tuple[str, float]:
        """Detect artifacts in the raw data."""

        labels = self.label_artifacts(extra_labels=self.session_triggers)
        percent_dropped = 0

        # calculate the impact of artifacts on the session triggers by creating epochs around the triggers
        # and dropping artifacts using MNE
        if self.session_triggers:
            log.info('Calculating the impact of artifacts on session triggers.')
            epochs = mne_epochs(
                self.mne_data,
                self.trial_duration,
                reject_by_annotation=True,
                preload=True)

            # calculate the percentage of dropped epochs
            kept_epochs = len(epochs)
            total_epochs = len(self.trigger_label)
            percent_dropped = (1 - (kept_epochs / total_epochs)) * 100
            self.dropped = (
                f'{kept_epochs}/{total_epochs} retained. {percent_dropped}% epochs dropped due to artifacts.'
            )
            log.info(f'{self.dropped=}')

        self.analysis_done = True
        if self.save_path:
            self.save_artifacts(overwrite=True)
            log.info(f'Artifact labelled data saved to {self.save_path}')

        labels = f'{len(labels)} artifacts found in the data.'

        return labels, percent_dropped

    def label_artifacts(
            self,
            detect_voltage: bool = True,
            detect_eog: bool = True,
            extra_labels: Optional[Annotations] = None) -> Annotations:
        """Label the artifacts in the raw data."""
        # Create an empty annotations object to store all the annotations
        annotations = mne.Annotations(0, 0, 'start')

        if detect_voltage:
            voltage = self.label_voltage_events()
            if voltage:
                voltage_annotations, bad_channels = voltage
                if bad_channels:
                    # add bad channel labels to the raw data
                    self.mne_data.info['bads'] = bad_channels
                    log.info(f'Bad channels detected: {bad_channels}')

                if voltage_annotations:
                    log.info(f'Voltage violation events found: {len(voltage_annotations)}')
                    annotations += voltage_annotations
                    self.voltage_annotations = voltage_annotations

        if detect_eog:
            eog = self.label_eog_events()
            if eog:
                eog_annotations, eog_events = eog

                if eog_annotations:
                    log.info(f'EOG events found: {len(eog_events)}')
                    annotations += eog_annotations
                    self.eog_annotations = eog_annotations

        # Combine the annotations if multiple returned because set_annotations overwrites
        # existing labels.
        if extra_labels:
            annotations += extra_labels

        if len(annotations) > 0:
            self.mne_data.set_annotations(annotations)
        else:
            log.info('No artifacts or labels provided for the data.')

        # Plot the data with all annotations. This allows a user to reject additional bad epochs
        # or modify existing annotations.
        if self.semi_automatic:
            self.mne_data.plot(block=True)

        return annotations

    def save_artifacts(self, overwrite: bool = False) -> None:
        """Save the artifact file to disk."""
        if self.analysis_done:
            self.mne_data.save(
                f'{self.save_path}/{DefaultArtifactParameters.ARTIFACT_LABELLED_FILENAME.value}',
                overwrite=overwrite)
        else:
            log.info('Artifact cannot be saved, artifact analysis has been done yet.')

    def raw_data_to_mne(self, raw_data: RawData, volts: bool = False) -> mne.io.RawArray:
        """Convert the raw data to an MNE RawArray."""

        downsample_rate = self.parameters.get("down_sampling_rate")
        notch_filter = self.parameters.get("notch_filter_frequency")
        filter_high = self.parameters.get("filter_high")
        filter_low = self.parameters.get("filter_low")
        filter_order = self.parameters.get("filter_order")

        default_transform = get_default_transform(
            sample_rate_hz=raw_data.sample_rate,
            notch_freq_hz=notch_filter,
            bandpass_low=filter_low,
            bandpass_high=filter_high,
            bandpass_order=filter_order,
            downsample_factor=downsample_rate,
        )

        return convert_to_mne(
            raw_data,
            channel_map=analysis_channels(raw_data.channels, self.device_spec),
            transform=default_transform,
            volts=volts)

    def label_eog_events(
            self,
            preblink: float = DefaultArtifactParameters.EOG_LABEL_DURATION.value,
            postblink: float = DefaultArtifactParameters.EOG_LABEL_DURATION.value,
            label: str = ArtifactType.BLINK.label(),
            threshold: float = DefaultArtifactParameters.EOG_THRESHOLD.value) -> Optional[Tuple[mne.Annotations, list]]:
        """Label EOG artifacts.

        Parameters
        ----------
        raw : mne.RawArray
        preblink : float | int
            The time before the blink to label as an artifact. Defaults to DefaultArtifactParameters.EOG_DURATION.
        postblink : float | int
            The time after the blink to label as an artifact. Defaults to DefaultArtifactParameters.EOG_DURATION.
        label : str
            The label to use for the eog epochs. Note: The prefix
                BAD or BAD_ must be present for epoch rejection. Defaults to ArtifactType.BLINK.label().
        threshold : float
            The voltage threshold to use for detecting blinks. Defaults to DefaultArtifactParameters.EOG_THRESHOLD.

        Returns
        -------
        ( mne.Annotations, list) | None
        """
        if not self.eye_channels:
            log.info('No eye channels provided. Cannot detect EOG artifacts.')
            return None

        log.info(f'Using blink threshold of {threshold} for channels {self.eye_channels}.')
        eog_events = mne.preprocessing.find_eog_events(self.mne_data, ch_name=self.eye_channels, thresh=threshold)
        # eog_events = mne.preprocessing.ica_find_eog_events(raw) TODO compare to ICA

        if len(eog_events) > 0:
            # Create annotations around blinks
            onsets = eog_events[:, 0] / self.mne_data.info['sfreq'] - preblink
            durations = [postblink + preblink] * len(eog_events)
            descriptions = [label] * len(eog_events)
            blink_annotations = mne.Annotations(onsets, durations, descriptions)

            return blink_annotations, eog_events
        return None

    def label_voltage_events(
            self,
            pre_event: float = DefaultArtifactParameters.VOlTAGE_LABEL_DURATION.value,
            post_event: float = DefaultArtifactParameters.VOlTAGE_LABEL_DURATION.value,
            peak: Optional[Tuple[float, float, str]] = None,
            flat: Optional[Tuple[float, float, str]] = None) -> Optional[Tuple[mne.Annotations, List[str]]]:
        """Annotate Voltage Events in the data.

        Parameters
        ----------
        raw : mne.RawArray
        pre_event : float | int
            The time before the voltage to label.
        post_event : float | int
            The time after the voltage to label.
        peak : tuple
            The voltage threshold to use for peak detection. The tuple should be (threshold, min_duration, label).
            Defaults to (
                DefaultArtifactParameters.PEAK_THRESHOLD.value,
                DefaultArtifactParameters.PEAK_MIN_DURATION.value,
                ArtifactType.VOLTAGE.label()
            )
        flat : tuple
            The voltage threshold to use for flat detection. The tuple should be (threshold, min_duration, label).
            Defaults to (
                DefaultArtifactParameters.FLAT_THRESHOLD.value,
                DefaultArtifactParameters.FLAT_MIN_DURATION.value,
                ArtifactType.FLAT.label())

        Returns
        -------
        (mne.Annotations, list) | None

        """
        if not flat:
            flat = (
                DefaultArtifactParameters.FLAT_THRESHOLD.value,
                DefaultArtifactParameters.FLAT_MIN_DURATION.value,
                ArtifactType.FLAT.label())

        if not peak:
            peak = (
                DefaultArtifactParameters.PEAK_THRESHOLD.value,
                DefaultArtifactParameters.PEAK_MIN_DURATION.value,
                ArtifactType.VOLTAGE.label())

        onsets: List[float] = []
        duration: List[float] = []
        descriptions: List[str] = []

        # get the voltage events for threshold
        peak_voltage_annotations, bad_channels1 = mne.preprocessing.annotate_amplitude(
            self.mne_data,
            min_duration=peak[1],
            bad_percent=self.percent_bad,
            peak=peak[0])
        if len(peak_voltage_annotations) > 0:
            log.info(f'Peak voltage events found: {len(peak_voltage_annotations)}')
            onsets, durations, descriptions = self.concat_annotations(
                peak_voltage_annotations,
                pre_event,
                post_event,
                peak[2],
                onsets,
                duration,
                descriptions)

        # get the voltage events for flat
        flat_voltage_annotations, bad_channels2 = mne.preprocessing.annotate_amplitude(
            self.mne_data, min_duration=flat[1], bad_percent=self.percent_bad, flat=flat[0])
        if len(flat_voltage_annotations) > 0:
            log.info(f'Flat voltage events found: {len(flat_voltage_annotations)}')
            onsets, durations, descriptions = self.concat_annotations(
                flat_voltage_annotations,
                pre_event,
                post_event,
                flat[2],
                onsets,
                duration,
                descriptions)

        # combine the bad channels
        bad_channels = bad_channels1 + bad_channels2

        if len(bad_channels) == 0:
            bad_channels = None

        if len(onsets) > 0:
            return mne.Annotations(onsets, durations, descriptions), bad_channels
        else:
            log.info('No voltage events found.')
            return None, bad_channels

    def concat_annotations(
            self,
            annotations: mne.Annotations,
            pre: Union[float, int],
            post: Union[float, int],
            label: str,
            onsets: List[float],
            duration: List[str],
            descriptions: List[str]) -> Tuple[list, list, list]:
        """Concatenate annotations.

        Iters through the annotations and adds the pre and post time to the onsets and duration.

        Parameters
        ----------
        annotations : mne.Annotations
        pre : float | int
            The time before the voltage to label.
        post : float | int
            The time after the voltage to label.
        label : str
            The label to use for the voltage epochs. Note: The prefix
                BAD or BAD_ must be present for epoch rejection.
        onsets : list
            The list of onsets to append to.
        duration : list
            The list of durations to append to.
        descriptions : list
            The list of descriptions to append to.

        Returns
        -------
        Tuple[list, list, list] - The onsets, durations, and descriptions.

        """

        for annotation in annotations:
            # hack for the start of the recording, if greater than prevoltage, then set to 0.0
            if (annotation['onset']) > pre:
                onset = annotation['onset'] - pre
            else:
                onset = 0.0
            onsets.append(onset)
            duration.append(annotation['duration'] + pre + post)
            descriptions.append(label)

        return onsets, duration, descriptions


def write_mne_annotations(
        annotations: mne.Annotations,
        path: str,
        filename: str) -> None:
    """Write MNE annotations to a text file.

    This is useful for sharing annotations for use outside of MNE.

    Parameters
    ----------
    annotations : mne.Annotations
        The annotations to write.
    path : str
        The path to write the annotations to.
    filename : str
        The filename for the new annotations file.

    Returns
    -------
    None
    """
    # write the annotations to a text file where each line is [description, bcipy_label, onset, duration]
    with open(os.path.join(path, filename), 'w') as tmp_file:
        for annotation in annotations:
            tmp_file.write(
                f"{annotation['description'].split('BAD_')[-1]} "
                f"{TriggerType.ARTIFACT} {annotation['onset']} {annotation['duration']}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the directory with >= 1 sessions to be analyzed for artifacts',
        required=False)
    # Semi-supervised artifact detection.
    # If flag used, the user will be prompted with a GUI to correct or add labels for artifacts.
    parser.add_argument("--semi", dest="semi", action="store_true")
    # Save the data after artifact detection as .fif file (MNE).
    parser.add_argument("--save", dest="save", action="store_true")
    # If flag used with semi, the user will be prompted to correct or add labels for
    # artifacts alongside the other session triggers.
    parser.add_argument("--colabel", dest="colabel", action="store_true")
    parser.set_defaults(semi=False)
    parser.set_defaults(save=False)
    parser.set_defaults(colabel=False)
    args = parser.parse_args()

    # if no path is provided, prompt for one using a GUI
    path = args.path
    if not path:
        path = load_experimental_data()

    positions = None
    for session in Path(path).iterdir():
        # loop through the sessions, pausing after each one to allow for manual stopping
        if session.is_dir():
            print(f'Processing {session}')
            prompt = input('Hit enter to continue or type "skip" to skip processing: ')
            if prompt != 'skip':
                # load the parameters from the data directory
                parameters = load_json_parameters(
                    f'{session}/{DEFAULT_PARAMETERS_FILENAME}', value_cast=True)

                # load the raw data from the data directory
                raw_data = load_raw_data(str(Path(session, f'{RAW_DATA_FILENAME}.csv')))
                type_amp = raw_data.daq_type

                # load the triggers
                if args.colabel:
                    trigger_type, trigger_timing, trigger_label = trigger_decoder(
                        offset=0.1,
                        trigger_path=f"{session}/{TRIGGER_FILENAME}",
                        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
                    )
                    triggers = (trigger_type, trigger_timing, trigger_label)
                else:
                    triggers = None

                devices.load(Path(BCIPY_ROOT, DEFAULT_DEVICE_SPEC_FILENAME))
                device_spec = devices.preconfigured_device(raw_data.daq_type)

                # check the device spec for any frontal channels to use for EOG detection
                eye_channels = []
                for channel in device_spec.channels:
                    if 'F' in channel:
                        eye_channels.append(channel)
                if len(eye_channels) == 0:
                    eye_channels = None

                artifact_detector = ArtifactDetection(
                    raw_data,
                    parameters,
                    device_spec,
                    eye_channels=eye_channels,
                    session_triggers=triggers,
                    save_path=None if not args.save else session,
                    semi_automatic=args.semi)

                detected = artifact_detector.detect_artifacts()
                # write_mne_annotations(detected, session, 'artifacts.txt')
            else:
                log.info(f'Skipping {session}')
            # Uncomment below to pause between sessions.
            # breakpoint()
