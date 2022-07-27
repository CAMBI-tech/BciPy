from bcipy.helpers.load import (
    load_raw_data,
)
from pathlib import Path


def main(data_folder):
    raw_data_file = 'raw_data.csv'
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate
    # access data using raw_data.channel_data or raw_data.numerical_data
    print('Successfully loaded raw data from {}'.format(raw_data_file))
    print('Channels: {}'.format(channels))
    print('Type: {}'.format(type_amp))
    print('Sample rate: {}'.format(sample_rate))
    print('Data shape: {}'.format(raw_data.channel_data.shape))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', required=True, help='Path to data folder')
    args = parser.parse_args()

    main(args.data_folder)
