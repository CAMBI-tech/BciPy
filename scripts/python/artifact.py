import mne
mne.set_log_level('WARNING')
import os
from pathlib import Path
from typing import Union, List, Tuple, Optional
from bcipy.config import DEFAULT_PARAMETER_FILENAME, RAW_DATA_FILENAME, TRIGGER_FILENAME, DEFAULT_DEVICE_SPEC_FILENAME, BCIPY_ROOT
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.system_utils import report_execution_time
from mne import Annotations
from bcipy.helpers.convert import convert_to_mne
from bcipy.helpers.load import load_raw_data
from bcipy.signal.process import get_default_transform
from bcipy.helpers.triggers import TriggerType, trigger_decoder
import bcipy.acquisition.devices as devices

ARTIFACT_LABELLED_FILENAME = 'auto_artifacts_raw.fif'

def artifact_rejection(
        path: str,
        percent_bad: float=50.0,
        semi_automatic_ar: bool=True,
        save_artifacts: bool=False,
        overwrite: bool=True,
        use_session_filter: bool=True,
        session_annotations: bool = False) -> mne.io.RawArray:
    """BciPy offline artifact rejection pipeline. 
    
    This pipeline is designed to be run after all data has been collected. It currently supports bcipy>=2.0.0rc1.
    
    The path provided should be a directory containing BciPy sessions. Each sub-directory should then contain
    the following files:
        - parameters.json
        - triggers.txt
        - raw_data.csv
    
    The pipeline can be run in semi-automatic mode, which will allow the user to manually inspect the data and reject
    bad epochs. This is useful for identifying and correcting artifacts that are not detected by the automatic methods.
    Semi-automatic mode is enabled by default. 

    Parameters
    ----------
    path : str
        The path to the data directory.
    percent_bad : float
        The percentage of bad epochs and channels to allow. Defaults to 50%.
    semi_automatic_ar : bool
        Whether to run the pipeline in semi-automatic mode. Defaults to True.
    save_artifacts : bool
        Whether to save the artifacts to disk. Defaults to False.
    overwrite : bool
        Whether to overwrite existing data. Defaults to True.
    use_session_filter : bool
        Whether to use the session filter. Defaults to True. TODO: implement custom filters.
    session_annotations : bool
        Whether to use the session annotations like target/nontarget. Defaults to False. 


    Returns 
    -------
    mne_data : mne.io.RawArray
        The raw data labelled with annotations for artifacts.
    """
    parameters = load_json_parameters(f'{path}/{DEFAULT_PARAMETER_FILENAME}', value_cast=True)

    # get raw data from session folder
    raw_data = load_raw_data(Path(path, f'{RAW_DATA_FILENAME}.csv'))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

    devices.load(Path(BCIPY_ROOT, DEFAULT_DEVICE_SPEC_FILENAME))
    device_spec = devices.preconfigured_device(raw_data.daq_type)


    # using this we remove channels unrelated to the task or that are not used in the analysis
    channel_map = analysis_channels(channels, device_spec)
    static_offset = parameters.get('static_trigger_offset')
    trial_length = parameters.get('trial_length')

    # get trigger information
    trigger_targetness, trigger_timing, _ = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{path}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )

    # trigger timing and labels
    if session_annotations:
        session_annotations = Annotations(trigger_timing, [trial_length] * len(trigger_timing), trigger_targetness)

    if use_session_filter:
        # get signal filtering information
        downsample_rate = parameters.get("down_sampling_rate")
        # downsample_rate = 1 # comment out if you want to use the downsample rate from the parameters file
        notch_filter = parameters.get("notch_filter_frequency")
        filter_high = parameters.get("filter_high")
        filter_low = parameters.get("filter_low")
        filter_order = parameters.get("filter_order")
        # setup filtering ... do not downsample!! We might want to do this after preprocessing
        default_transform = get_default_transform(
            sample_rate_hz=sample_rate,
            notch_freq_hz=notch_filter,
            bandpass_low=filter_low,
            bandpass_high=filter_high,
            bandpass_order=filter_order,
            downsample_factor=downsample_rate,
        )

        mne_data, _ = convert_to_mne(raw_data, channel_map=channel_map, transform=default_transform, volts=False)
    else:
        # Custom filters can be applied here. https://mne.tools/stable/generated/mne.filter.filter_data.html
        mne_data, _ = convert_to_mne(raw_data, channel_map=channel_map, volts=False) # don't apply the session filters

    # artifact handling pipeline
    mne_data = semi_automatic_artifact_rejection(mne_data, percent_bad=percent_bad, verify_plot=semi_automatic_ar, extra_labels=session_annotations)

    # if desired, save the artifact labelled data as an MNE fif file and export the annotations as a text file
    if save_artifacts:
        mne_data.save(f'{path}/{ARTIFACT_LABELLED_FILENAME}', overwrite=overwrite)
        write_mne_annotations(mne_data.annotations, path)

    return mne_data

def semi_automatic_artifact_rejection(
        mne_data: mne.io.RawArray,
        percent_bad: float=50.0,
        verify_plot: bool=True,
        extra_labels: list = []) -> mne.io.RawArray:
    """Semi-automatic artifact rejection.

    Parameters
    ----------
    mne_data : mne.RawArray
        The raw data to process for artifacts.
    percent_bad : float
        The percentage of bad epochs to allow. Defaults to 50%.
    verify_plot : bool
        Whether to plot the data before returning. This allows for semi-automatic artifact rejection.
            Set to false to reject by detection only, no human validation. Default True.
        Standard of Practice: 
            + Correct bad labels, but leave overlapping events!
            + Modify the duration of the events as needed
            + Labelling bad channels: a visual inspection of the channel in relation to all the noise present 
        Labels:
            BAD_peak: pop to electrical artifact; high amplitude artifact
            BAD_blink: eye blink activity
            BAD_eog: eye movement activity
            BAD_emg: muscle activity
            BAD_ecg: heart activity
            BAD_flat: see rules for flat
            BAD_event: we are not sure what this is, but noise is present

    Returns
    -------
    mne_data : mne.io.RawArray
        The raw data labelled with annotations for artifacts.
    """
    # add any extra labels to the data in case no artifacts are detected
    annotations = Annotations(0, 0, 'ignore')

    # VOLTAGE: here we rely on the defaults below for thresholding. See label_voltage_events for more info.
    voltage = label_voltage_events(
        mne_data,
        bad_percent_per_channel=percent_bad)
    if voltage:
        voltage_annotations, bad_channels = voltage
        print(f'Voltage violation events found: {len(voltage_annotations)}')
        if bad_channels:
            # add bad channel labels to the raw data
            mne_data.info['bads'] = bad_channels
            print(f'Bad channels detected: {bad_channels}')
        annotations += voltage_annotations

    # EOG: here we rely on the defaults below for thresholding. See label_eog_events for more info.
    eog = label_eog_events(mne_data)
    if eog:
        eog_annotations, eog_events = eog
        print(f'EOG events found: {len(eog_events)}')
        annotations += eog_annotations

    # combine the annotations here if both returned because set_annotations overwrites
    # and we are not guaranteed any annotations
    if extra_labels:
        annotations += extra_labels

    # set the annotations on the raw data
    mne_data.set_annotations(annotations)

    # plot the data with annotations. This allows the user to reject bad epochs and correct annotations.
    if verify_plot:
        mne_data.plot(block=True)

    return mne_data

@report_execution_time
def label_eog_events(
        raw: mne.io.RawArray,
        preblink: float=.5,
        postblink: float=.5,
        label: str='BAD_blink',
        threshold: float=80e-6,
        eye_channels: List[str]=['Fp1', 'Fp2', 'F7', 'F8']) -> Optional[Tuple[mne.Annotations, list]]:
    """Label EOG artifacts.

    Parameters
    ----------
    raw : mne.RawArray
    preblink : float | int
        The time before the blink to label as an artifact.
    postblink : float | int
        The time after the blink to label as an artifact.
    label : str
        The label to use for the eog epochs. Note: The prefix
            BAD or BAD_ must be present for epoch rejection.
    threshold : float
        The voltage threshold to use for detecting blinks.
    eye_channels : list
        The channels to use for detecting blinks.

    Returns
    -------
    ( mne.Annotations, list) | None
    """
    print(f'Using blink threshold of {threshold}')
    eog_events = mne.preprocessing.find_eog_events(raw, ch_name=eye_channels, thresh=threshold)
    # eog_events = mne.preprocessing.ica_find_eog_events(raw)

    if len(eog_events) > 0:
        # Create annotations around blinks
        onsets = eog_events[:, 0] / raw.info['sfreq'] - preblink
        durations = [postblink + preblink] * len(eog_events)
        descriptions = [label] * len(eog_events)
        blink_annotations = mne.Annotations(onsets, durations, descriptions)

        return blink_annotations, eog_events
    return None

def label_voltage_events(
        raw: mne.io.RawArray,
        pre_event: float=0.25,
        post_event: float=0.25,
        threshold: Tuple[float, float, str]=(50e-6, 0.001, 'BAD_peak'),
        flat: Tuple[float, float, str]= (.5e-6, 0.1, 'BAD_flat'),
        bad_percent_per_channel: Union[float, int]=50.0) -> Optional[Tuple[mne.Annotations, List[str]]]:
    """Annotate Voltage Events in the data.

    Parameters
    ----------
    raw : mne.RawArray
    pre_event : float | int
        The time before the voltage to label.
    post_event : float | int
        The time after the voltage to label.
    threshold : tuple
        The voltage threshold to use for peak detection. The tuple should be (threshold, min_duration, label).
    flat : tuple
        The voltage threshold to use for flat detection. The tuple should be (threshold, min_duration, label).
    bad_percent_per_channel : float | int
        The percentage of bad epochs to allow per channel.

    Returns
    -------
    (mne.Annotations, list) | None

    """
    onsets = []
    duration = []
    descriptions = []

    # get the voltage events for threshold
    peak_voltage_annotations, bad_channels1 = mne.preprocessing.annotate_amplitude(
        raw,
        min_duration=threshold[1],
        bad_percent=bad_percent_per_channel,
        peak=threshold[0])
    if len(peak_voltage_annotations) > 0:
        concat_annotations(
            peak_voltage_annotations,
            pre_event,
            post_event,
            threshold[2],
            onsets,
            duration,
            descriptions)
    
    # get the voltage events for flat
    flat_voltage_annotations, bad_channels2 = mne.preprocessing.annotate_amplitude(
        raw,
        min_duration=flat[1],
        bad_percent=bad_percent_per_channel,
        flat=flat[0])
    if len(flat_voltage_annotations) > 0:
        concat_annotations(
            flat_voltage_annotations,
            pre_event,
            post_event,
            flat[2],
            onsets,
            duration,
            descriptions)

    # combine the bad channels
    bad_channels = bad_channels1 + bad_channels2

    if len(onsets) > 0 or len(bad_channels) > 0:
        return mne.Annotations(onsets, duration, descriptions), bad_channels

    return None

def concat_annotations(
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
        filename: str='mne_annotations.txt') -> None:
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
            tmp_file.write(f"{annotation['description'].split('BAD_')[-1]} {TriggerType.ARTIFACT} {annotation['onset']} {annotation['duration']}\n")
    

if __name__ == "__main__":
    # """
    # To use, 
    #     1. Create a virtual environment with the requirements listed in requirements.txt. The bcipy version will determine what
    #         python version you need installed. Currently, bcipy>=2.0.0rc1 requires python>=3.7 and <3.9.
    #     2. Run the script. `python artifact.py`. 
    #         By default, it will prompt you for a data directory. Detect artifact automatically and quit.
    #     3. Run the script again using available flags. 
    #         `-p <path to data directory> "C:\Users\user\data\"`
    #         `--save <save data as .fif after detection for reuse later>`
    #         `--semi <semi-automatic mode with gui for artifact label editing>`
    #         `--colabel <colabel data with session annotations [target/nontarget]>`
    # """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the directory with raw_data to be converted',
        required=False)
    # Semi-supervised artifact detection. 
    # If flag used, the user will be prompted with a GUI to correct or add labels for artifacts.
    parser.add_argument("--semi", dest="semi", action="store_true")
    # Save the data after artifact detection as .fif file.
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

            prompt = input(f'Hit enter to continue or type "skip" to skip processing: ')
            if prompt != 'skip':
                mne_data = artifact_rejection(
                        session,
                        semi_automatic_ar=args.semi,
                        save_artifacts=args.save,
                        session_annotations=args.colabel,
                        overwrite=True)
            else:
                print(f'Skipping {session}')
            import pdb; pdb.set_trace() # comment out to continue without stopping between sessions