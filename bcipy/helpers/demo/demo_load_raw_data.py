from bcipy.helpers.load import (
    load_raw_data, load_json_parameters
)
from pathlib import Path
from bcipy.config import (DEFAULT_PARAMETERS_PATH, EYE_TRACKER_FILENAME_PREFIX)


def main(data_folder, data_file_paths):
    """Load BciPy raw data, print info to console, and return data to caller."""
    # Load all data saved in the directory

    raw_data_list = load_raw_data(data_folder, data_file_paths)
    for mode_data in raw_data_list:
        channels = mode_data.channels
        type_amp = mode_data.daq_type
        sample_rate = mode_data.sample_rate
        # access data using raw_data.channel_data or raw_data.numerical_data
        print(f'Successfully loaded raw data from {type_amp}')
        print(f'Channels: {channels}')
        print(f'Type: {type_amp}')
        print(f'Sample rate: {sample_rate}')
        print(f'Data shape: {mode_data.channel_data.shape}')
    return raw_data_list


if __name__ == "__main__":
    import argparse
    from bcipy.config import RAW_DATA_FILENAME

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', required=True, help='Path to data folder')
    parser.add_argument("-p", "--parameters_file", default=DEFAULT_PARAMETERS_PATH)
    args = parser.parse_args()

    parameters = load_json_parameters(args.parameters_file, value_cast=True)
    acq_mode = parameters.get("acq_mode")

    data_file_paths = []
    if '+' in acq_mode:
        if 'eyetracker' in acq_mode.lower():
            # find the eyetracker data file with the prefix of EYE_TRACKER_FILENAME_PREFIX
            eye_tracker_file = [f.name for f in Path(args.data_folder).iterdir()
                                if f.name.startswith(EYE_TRACKER_FILENAME_PREFIX)]
            assert len(
                eye_tracker_file) == 1, f"Found {len(eye_tracker_file)} eyetracker files in {args.data_folder}."
            data_file_paths.extend(eye_tracker_file)
        else:
            raise ValueError(f"Unsupported acquisition mode: {acq_mode}. Eyetracker must be included.")
        if 'eeg' in acq_mode.lower():
            # find the eeg data file with the prefix of EEG_FILENAME_PREFIX
            eeg_file = [f.name for f in Path(args.data_folder).iterdir() if f.name.startswith(RAW_DATA_FILENAME)]
            assert len(eeg_file) == 1, f"Found {len(eeg_file)} EEG files in {args.data_folder}. Expected 1."
            data_file_paths.extend(eeg_file)
        else:
            raise ValueError(f"Unsupported acquisition mode: {acq_mode}. EEG must be included.")
    else:
        data_file_paths = [f"{RAW_DATA_FILENAME}.csv"]

    raw_data_list = main(args.data_folder, data_file_paths)
