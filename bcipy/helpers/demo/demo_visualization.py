"""Demonstrates BciPy visualize ERP functionality!

To use at bcipy root,

    - `python bcipy/helpers/demo/demo_visualization.py"`:
        this will prompt for a BciPy data folder with raw_data.csv, parameters.json and triggers.txt

    - `python bcipy/helpers/demo/demo_visualization.py -p "path://to/bcipy/data/folder"`

    - `python bcipy/helpers/demo/demo_visualization.py --show"`
        this will show the visualization generated in closeable windows

    - - `python bcipy/helpers/demo/demo_visualization.py --save"`
        this will save the visualizations generated to the provided or selected path
"""
from bcipy.config import (TRIGGER_FILENAME, RAW_DATA_FILENAME, EYE_TRACKER_FILENAME_PREFIX,
                          DEFAULT_DEVICE_SPEC_FILENAME, DEFAULT_PARAMETERS_PATH,
                          DEFAULT_PARAMETER_FILENAME)
from bcipy.helpers.acquisition import analysis_channels
from bcipy.signal.process import get_default_transform
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.visualization import visualize_erp
from bcipy.helpers.triggers import TriggerType, trigger_decoder
import bcipy.acquisition.devices as devices


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_folder',
        help='Path to the directory with raw_data to be converted',
        required=False)
    parser.add_argument("-p", "--parameters_file", default=DEFAULT_PARAMETERS_PATH)
    parser.add_argument("--show", dest="show", action="store_true")
    parser.add_argument("--save", dest="save", action="store_true")
    parser.set_defaults(show=False)
    parser.set_defaults(save=False)
    args = parser.parse_args()

    path = args.data_folder
    if not path:
        path = load_experimental_data()
        parameters = load_json_parameters(f'{path}/{DEFAULT_PARAMETER_FILENAME}', value_cast=True)
    else:
        parameters = load_json_parameters(args.parameters_file, value_cast=True)

    acq_mode = parameters.get("acq_mode")

    # Make sure you are loading the correct raw data:
    data_file_paths = []
    if '+' in acq_mode:
        if 'eyetracker' in acq_mode.lower():
            # find the eyetracker data file with the prefix of EYE_TRACKER_FILENAME_PREFIX
            eye_tracker_file = [f.name for f in Path(path).iterdir()
                                if f.name.startswith(EYE_TRACKER_FILENAME_PREFIX)]
            assert len(
                eye_tracker_file) == 1, f"Found {len(eye_tracker_file)} eyetracker files in {path}. Expected 1."
            data_file_paths.extend(eye_tracker_file)
        else:
            raise ValueError(f"Unsupported acquisition mode: {acq_mode}. Eyetracker must be included.")
        if 'eeg' in acq_mode.lower():
            # find the eeg data file with the prefix of EEG_FILENAME_PREFIX
            eeg_file = [f.name for f in Path(path).iterdir() if f.name.startswith(RAW_DATA_FILENAME)]
            assert len(eeg_file) == 1, f"Found {len(eeg_file)} EEG files in {path}. Expected 1."
            data_file_paths.extend(eeg_file)
        else:
            raise ValueError(f"Unsupported acquisition mode: {acq_mode}. EEG must be included.")
    else:
        data_file_paths = [f"{RAW_DATA_FILENAME}.csv"]

    raw_data_list = load_raw_data(path, data_file_paths)

    for mode_data in raw_data_list:
        devices.load(Path(path, DEFAULT_DEVICE_SPEC_FILENAME))
        device_spec = devices.preconfigured_device(mode_data.daq_type)
        # extract  all relevant parameters from raw data object eeg
        if device_spec.content_type == "Eyetracker":
            continue
        elif device_spec.content_type == "EEG":
            trial_window = parameters.get("trial_window")
            if trial_window is None:
                trial_window = [0.0, 0.5]
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
            static_offset = parameters.get("static_trigger_offset")
            channels = mode_data.channels
            type_amp = mode_data.daq_type
            sample_rate = mode_data.sample_rate

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
            labels = [0 if label == 'nontarget' else 1 for label in trigger_targetness]

            channel_map = analysis_channels(channels, device_spec)

            save_path = None if not args.save else path

            figure_handles = visualize_erp(
                mode_data,
                channel_map,
                trigger_timing,
                labels,
                trial_window,
                transform=default_transform,
                plot_average=True,
                plot_topomaps=True,
                save_path=save_path,
                show=args.show
            )
