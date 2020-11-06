"""Demonstrates converting raw_data output to other EEG formats"""
from bcipy.helpers.convert import convert_to_edf
from mne.io import read_raw_edf


def plot_edf(edf_path: str, auto_scale: bool = False):
    """Plot data from the raw edf file. Note: this works from an iPython
    session but seems to throw errors when provided in a script.
    
    Parameters
    ----------
        edf_path - full path to the generated edf file
        auto_scale - optional; if True will scale the EEG data; this is
            useful for fake (random) data but makes real data hard to read.
    """
    edf = read_raw_edf(edf_path, preload=True)
    if auto_scale:
        edf.plot(scalings='auto')
    else:
        edf.plot()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the directory with raw_data to be converted',
        required=True)
    args = parser.parse_args()
    edf_path = convert_to_edf(args.path)
    print(f"\nWrote edf file to {edf_path}")
