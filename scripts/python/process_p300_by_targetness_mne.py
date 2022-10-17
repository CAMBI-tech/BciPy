"""Analysis Notes:

This script processes a folder of session data, looping over each session and processing the data into an average of epochs by poisition accross subjects and trials.

Filtered: Yes (default) 1-20 Hz, 2nd order butterworth, 60 Hz notch filter
Baseline: -50ms to 0ms
Epochs: 0ms to 500ms
Artifact Rejection: Semi-automatic (no more than 10% bad epochs per condition)

Trial Inclusion:
- free from artifacts: no eye-blinks, no muscle artifacts, no voltage violations ()
    Report on where the artifacts tend to be? Is it true the last one is most likely to be contain artifacts due to blinking?
"""
import mne
mne.set_log_level('WARNING')
from pathlib import Path
from typing import Union
from bcipy.config import DEFAULT_PARAMETER_FILENAME, TRIGGER_FILENAME, RAW_DATA_FILENAME
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.visualization import visualize_evokeds, visualize_joint_average
from bcipy.helpers.convert import convert_to_mne
from bcipy.helpers.load import load_raw_data
from bcipy.helpers.stimuli import mne_epochs
from bcipy.signal.process import get_default_transform
from bcipy.helpers.triggers import TriggerType, trigger_decoder


def main(path, percent_bad=10.0, semi_automatic_ar=True, save_artifacts=False):
    parameters = load_json_parameters(f'{path}/{DEFAULT_PARAMETER_FILENAME}', value_cast=True)

    # extract all relevant parameters
    poststim_length = parameters.get("trial_length")
    # prestim_length = parameters.get("prestim_length") # used for online filtering
    trials_per_inquiry = parameters.get("stim_length")
    trials = parameters.get("stim_number")
    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    filter_high = parameters.get("filter_high")
    filter_low = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")

    # get raw data 
    raw_data = load_raw_data(Path(path, f'{RAW_DATA_FILENAME}.csv'))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

    # setup filtering ... do not downsample!! We might want to do this after preprocessing
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=notch_filter,
        bandpass_low=filter_low,
        bandpass_high=filter_high,
        bandpass_order=filter_order,
        downsample_factor=1,
    )

    # process triggers.txt files
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{path}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )

    labels = [0 if label == 'nontarget' else 1 for label in trigger_targetness]
    channel_map = analysis_channels(channels, type_amp)

    mne_data = convert_to_mne(raw_data, channel_map=channel_map, transform=default_transform)
    # mne_data = convert_to_mne(raw_data, channel_map=channel_map) # don't apply the filters

    # artifact handling pipeline
    mne_data = semi_automatic_artifact_rejection(mne_data, percent_bad=percent_bad, plot=semi_automatic_ar)

    if save_artifacts:
        mne_data.save(f'{path}/artifacts_raw.fif', overwrite=True)

    epochs = mne_epochs(
        mne_data,
        trigger_timing,
        poststim_length,
        labels,
        baseline=(-0.2, 0), # note baseline here!
        reject_by_annotation=True,
        channels=None)
    epochs.drop_bad()

    clases = [1, 2]
    permitted_loss = trials * (1 - (percent_bad / 100))
    for pos in clases:
        epoch_pos = epochs[f'{pos}']
        print(f'Class {pos} has {len(epoch_pos)} epochs')
        if len(epoch_pos) < permitted_loss:
            print('Too many epochs rejected. This session will be excluded from analysis.')
            return None

    return epochs, clases, mne_data

def semi_automatic_artifact_rejection(mne_data, percent_bad=5.0, plot=False):
    """Semi-automatic artifact rejection.

    Parameters
    ----------
    mne_data : mne.RawArray
        The raw data to process for artifacts.
    percent_bad : float
        The percentage of bad epochs to allow.
    plot : bool
        Whether to plot the data before returning.

    Returns
    -------
    mne_data : mne.RawArray
        The raw data labelled with annotations for artifacts.
    """
    voltage = label_voltage_events(mne_data, bad_percent_per_channel=percent_bad)
    if voltage:
        voltage_annotations, bad_channels = voltage
        print(f'Voltage violation events found: {len(voltage_annotations)}')
        if bad_channels:
            # add bad channel labels to the raw data
            mne_data.info['bads'] = bad_channels
            print(f'Bad channels detected: {bad_channels}')
        mne_data.set_annotations(voltage_annotations)
    eog = label_eog_events(mne_data)
    if eog:
        eog_events, eog_annotations = eog
        print(f'EOG events found: {len(eog_events)}')
        mne_data.set_annotations(eog_annotations)

    if voltage and eog:
        # combine the annotations here beceause set_annotations overwrites and we are not garunteed any annotations
        mne_data.set_annotations(voltage_annotations + eog_annotations)

    # plot the data with annotations and allow the user to reject bad epochs
    if plot:
        mne_data.plot(block=True)
    return mne_data

def semi_automatic_ica(data, n_components=0.99, method='fastica'):
    """Semi-automatic ICA artifact detection and rejection.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    n_components : int | float
        The number of components to use. If float, the number of components
        is estimated from the data.
    method : str
        The ICA method to use. Defaults to 'fastica'.

    Returns
    -------
    epochs : mne.Epochs
        The epochs with bad epochs removed.
    """
    ica = mne.preprocessing.ICA(n_components=n_components, method=method)
    ica.fit(data)
    ica.plot_components()
    ica.apply(data)
    return data, ica

def label_eog_events(
        raw,
        preblink=.5,
        postblink=.5,
        label='BAD blink',
        eye_channels=['Fp1', 'Fp2', 'F7', 'F8']):
    """Label EOG artifacts.

    Parameters
    ----------
    raw : mne.RawArray
    preblink : float | int
        The time before the blink to label as an artifact.
    postblink : float | int
        The time after the blink to label as an artifact.
    label : str
        The label to use for the eog epochs.

    Returns
    -------
    (mne.Events, mne.Annotations) | None
    """
    # find a good blink threshold for this person based on the max/min of the eye channels
    data, _ = raw[eye_channels[0]]
    threshold = 100e-6
    print(f'Using blink threshold of {threshold}')
    eog_events = mne.preprocessing.find_eog_events(raw, ch_name=eye_channels, thresh=threshold)

    if len(eog_events) > 0:
        # Create annotations around blinks
        onsets = eog_events[:, 0] / raw.info['sfreq'] - preblink
        durations = [postblink + preblink] * len(eog_events)
        descriptions = [label] * len(eog_events)
        blink_annotations = mne.Annotations(onsets, durations, descriptions)

        return eog_events, blink_annotations
    return None

def label_voltage_events(
        raw: mne.io.RawArray,
        preemg: float=0.25,
        postemg: float=0.25,
        label: str='BAD peak',
        min_duration: float=0.05,
        threshold: float=125e-6,
        flat: float=.5e-6,
        bad_percent_per_channel: Union[float, int]=15.0):
    """Label EMG artifacts

    Parameters
    ----------
    raw : mne.RawArray
    preemg : float | int
        The time before the emg to label as an artifact.
    postemg : float | int
        The time after the emg to label as an artifact.
    label : str
        The label to use for the emg epochs.
    min_duration : float | int
        The minimum duration of an emg event.
    threshold : float | int
        The voltage threshold to use for emg detection.
    flat : float | int
        The flat threshold to use for emg detection.
    bad_percent_per_channel : float | int
        The percentage of bad epochs to allow per channel.

    Returns
    -------
    (mne.Annotations, list) | None

    """
    emg_annotations, bad_channels = mne.preprocessing.annotate_amplitude(
        raw,
        min_duration=min_duration,
        bad_percent=bad_percent_per_channel,
        peak=threshold,
        flat=flat)

    # Create annotations around EMG annotations returned including the pre/post emg time
    if len(emg_annotations) > 0:
        onsets = []
        duration = []
        descriptions = []
        for annotation in emg_annotations:
            # hack for the start of the recording, if greater than preemg, then set to 0.0
            if annotation['onset'] > preemg:
                annotation['onset'] -= preemg
            else:
                annotation['onset'] = 0.0
            onsets.append(annotation['onset'])
            duration.append(annotation['duration'] + preemg + postemg)
            descriptions.append(label)
        emg_annotations = mne.Annotations(onsets, duration, descriptions)
        return emg_annotations, bad_channels
    return None

def determine_p300_amplitude(epochs, conditions):
    """Determine the amplitude of the P300 response.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    conditions : list
        The conditions to process for amplitude differences.

    Returns
    -------
    p300_amplitude : float
        The amplitude of the P300 response.
    p300_latency : float
        The latency of the P300 response.
    p300_max : float
        The maximum voltage channel location of the P300 response.
    """
    p300_amplitudes = []
    p300_latency = []
    p300_maximal_location = []
    for con in conditions:
        average_per_condition = epochs[con].average()
        ch_name, latency, amplitude = average_per_condition.get_peak(
            mode='pos', return_amplitude=True, tmin=0.2, tmax=0.5)
        p300_amplitudes.append(amplitude)
        p300_latency.append(latency)
        p300_maximal_location.append(ch_name)
    return p300_amplitudes, p300_latency, p300_maximal_location

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the directory with raw_data to be converted',
        required=False)
    args = parser.parse_args()

    path = args.path
    if not path:
        path = load_experimental_data()
    
    all_epochs = []
    all_mne_data = []
    all_excluded = []
    positions = None
    for session in Path(path).iterdir():
        if session.is_dir():
            print(f'Processing {session}')

            resp = main(session, semi_automatic_ar=True)
            if resp:
                epochs, positions, mne_data = resp
                # mne_data.save(f'{session}/mne_annotated_eeg.fif', overwrite=True)
                all_epochs.append(epochs)
                all_mne_data.append(mne_data)
                epochs.save(f'{session}/target_nontarget_epochs.fif', overwrite=True)
            else:
                all_excluded.append(session)
                print(f'Excluding {session}')
        
        import pdb; pdb.set_trace()
    
    amps = determine_p300_amplitude(all_epochs, positions)

    # Grand averaging
    concat_epochs = []
    for position in positions:
        concat_epochs.append(
            mne.concatenate_epochs([epoch[f'{position}'] for epoch in all_epochs])
        )

    visualize_evokeds(concat_epochs, show=True)
    visualize_joint_average(concat_epochs, ['NonTarget', 'Target'], show=True)
    import pdb; pdb.set_trace()