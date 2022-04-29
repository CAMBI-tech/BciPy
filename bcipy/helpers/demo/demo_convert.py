"""Demonstrates converting BciPy data output to other EEG formats.

To use at bcipy root,

    `python bcipy/helpers/demo/demo_convert.py -p "path://to/bcipy/data/folder"`
"""
from bcipy.helpers.convert import convert_to_edf
from bcipy.helpers.visualization import plot_edf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the directory with raw_data to be converted',
        required=True)
    args = parser.parse_args()

    path = args.path
    edf_path = convert_to_edf(
        path,
        use_event_durations=True,
        write_targetness=False,
        overwrite=True,
        annotation_channels=None)
    plot_edf(edf_path)  # comment if not in an iPython notebook to plot using MNE
    print(f"\nWrote edf file to {edf_path}")
