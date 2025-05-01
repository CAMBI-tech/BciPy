# load all calibration files for RSVP and Matrix into MNE. Create group average of all users and save to output directory.
from pathlib import Path

import bcipy.acquisition.devices as devices
from bcipy.config import (DEFAULT_DEVICE_SPEC_FILENAME,
                          DEFAULT_PARAMETERS_FILENAME, RAW_DATA_FILENAME,
                          TRIGGER_FILENAME)
from bcipy.helpers.acquisition import analysis_channels
from bcipy.io.load import (load_experimental_data, load_json_parameters,
                           load_raw_data)
from bcipy.io.convert import convert_to_mne
from bcipy.core.stimuli import mne_epochs
from bcipy.core.triggers import TriggerType, trigger_decoder
from bcipy.helpers.visualization import visualize_erp
from bcipy.signal.process import get_default_transform
import mne


def load_bcipy_data(path):
    """Load BciPy data from the specified path."""
    parameters = load_json_parameters(f'{path}/{DEFAULT_PARAMETERS_FILENAME}',
                                      value_cast=True)

    # extract all relevant parameters
    trial_window = (0.0, 0.8)
    prestim_length = parameters.get("prestim_length")
    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)
    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    filter_high = parameters.get("filter_high")
    filter_low = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")

    raw_data = load_raw_data(Path(path, f'{RAW_DATA_FILENAME}.csv'))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

    devices.load(Path(path, DEFAULT_DEVICE_SPEC_FILENAME))
    device_spec = devices.preconfigured_device(raw_data.daq_type)
    static_offset = 0.09

    # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=notch_filter,
        bandpass_low=filter_low,
        bandpass_high=filter_high,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    # Process triggers.txt files
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{path}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )
    trigger_timing = [timing + trial_window[0] for timing in trigger_timing]
    labels = [0 if label == 'nontarget' else 1 for label in trigger_targetness]

    channel_map = analysis_channels(channels, device_spec)
    return raw_data, trigger_timing, labels, channel_map, default_transform
    
def concat_mne_epochs(mne_epochs_list):
    """Concatenate MNE epochs."""
    mne_epochs = mne.concatenate_epochs(mne_epochs_list)
    return mne_epochs

if __name__ == "__main__":
    
    PARADIGMS = ["RSVP", "Matrix"]
    task_type = "Calibration"
    trial_length = 0.8
    # Load the experimental data for the user
    # data_path = load_experimental_data()
    data_path = Path("C:\\Users\\tabme\\Desktop\\Mental Effort EEG recordings")

    print(f"Loading data from {data_path}")

    all_data = {}
    for user in data_path.iterdir():
        all_data[user.name] = {}
        for run in user.iterdir():
           for para in PARADIGMS:
                if para in run.name and task_type in run.name:
                    # assert the para is not in all_data[user.name]
                    if para in all_data[user.name]:
                        raise ValueError(f"Paradigm {para} already exists for user {user.name}")
                    all_data[user.name][para] = {}
                    raw_data, trigger_timing, labels, channel_map, transform = load_bcipy_data(run)
                    # convert to mne
                    all_data[user.name][para]["mne_raw"] = convert_to_mne(raw_data, channel_map=channel_map, transform=transform)
                    all_data[user.name][para]["mne_epochs"] = mne_epochs(
                        all_data[user.name][para]["mne_raw"], trial_length, trigger_timing, labels)
    
    # breakpoint()
    # Create a list of MNE epochs for paradigm
    rsvp_mne_epochs_list = []
    matrix_mne_epochs_list = []
    for user, data in all_data.items():
        if "RSVP" in data:
            rsvp_mne_epochs_list.append(data["RSVP"]["mne_epochs"])
        if "Matrix" in data:
            matrix_mne_epochs_list.append(data["Matrix"]["mne_epochs"])
    
    # Concatenate MNE epochs for each paradigm
    rsvp_mne_epochs = concat_mne_epochs(rsvp_mne_epochs_list)
    matrix_mne_epochs = concat_mne_epochs(matrix_mne_epochs_list)

    # Create group average for RSVP and Matrix
    rsvp_group_average_tgt = rsvp_mne_epochs['2']
    rsvp_group_average_nontgt = rsvp_mne_epochs['1']

    matrix_group_average_tgt = matrix_mne_epochs['2']
    matrix_group_average_nontgt = matrix_mne_epochs['1']

    breakpoint()
    plot_joint_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # plot the target/non-target ERPs for RSVP and Matrix
    rsvp_group_average_tgt.average().plot_joint(title="RSVP Target Group Average", times=plot_joint_times)
    rsvp_group_average_nontgt.average().plot_joint(title="RSVP Non-Target Group Average", times=plot_joint_times)

    matrix_group_average_tgt.average().plot_joint(title="Matrix Target Group Average", times=plot_joint_times)
    matrix_group_average_nontgt.average().plot_joint(title="Matrix Non-Target Group Average", times=plot_joint_times)


    # breakpoint()
    # using the plot compare evoked function to plot the group average
    # for RSVP and Matrix
    # We use the it
    evokeds = dict(nontarget=list(rsvp_group_average_nontgt.iter_evoked()),
                   target=list(rsvp_group_average_tgt.iter_evoked()))
    mne.viz.plot_compare_evokeds(
        evokeds,
        picks="eeg",
        title="RSVP Group Average Comparison",
        show_sensors=True,
        legend=True,
        show=True,
        combine="mean",
        ci=0.95,
    )

    evokeds = dict(nontarget=list(matrix_group_average_nontgt.iter_evoked()),
                   target=list(matrix_group_average_tgt.iter_evoked()))
    mne.viz.plot_compare_evokeds(
        evokeds,
        picks="eeg",
        title="Matrix Group Average Comparison",
        show_sensors=True,
        legend=True,
        show=True,
        combine="mean",
        ci=0.95,
    )

    
    