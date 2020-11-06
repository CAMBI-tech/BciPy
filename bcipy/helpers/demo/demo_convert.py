"""Demonstrates converting raw_data output to other EEG formats"""
from bcipy.helpers.convert import write_edf
from mne.io import read_raw_edf


def plot_edf(edf_path: str):
    """Plot data from the raw edf file. Note: this works from an iPython
    session but seems to throw errors when provided in a script."""
    edf = read_raw_edf(edf_path, preload=True)
    edf.plot(scalings='auto')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the directory with raw_data to be converted',
        required=True)
    args = parser.parse_args()
    edf_path = write_edf(args.path)
    print(f"\nWrote edf file to {edf_path}.")
