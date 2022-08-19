from bcipy.helpers.load import (
    load_raw_data,
)
from pathlib import Path


def main(data_folder, file_name):
    """Load BciPy raw data, print info to console, and return data to caller."""
    raw_data = load_raw_data(Path(data_folder, file_name))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate
    # access data using raw_data.channel_data or raw_data.numerical_data
    print(f'Successfully loaded raw data from {file_name}')
    print(f'Channels: {channels}')
    print(f'Type: {type_amp}')
    print(f'Sample rate: {sample_rate}')
    print(f'Data shape: {raw_data.channel_data.shape}')
    return raw_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', required=True, help='Path to data folder')
    args = parser.parse_args()

    raw_data = main(args.data_folder, file_name='raw_data.csv')
