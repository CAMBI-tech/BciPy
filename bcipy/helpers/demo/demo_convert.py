"""Demonstrates converting BciPy data output to other EEG formats.

To use at bcipy root,

    `python bcipy/helpers/demo/demo_convert.py -d "path://to/bcipy/data/folder"`
"""
from bcipy.helpers.convert import convert_to_edf, convert_to_bdf
from bcipy.helpers.load import load_experimental_data
from bcipy.helpers.visualization import plot_edf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--directory',
        help='Path to the directory with raw_data to be converted',
        required=False)
    parser.add_argument(
        '-p',
        '--plot',
        help='whether to plot the resulting edf file',
        required=False
    )
    args = parser.parse_args()

    path = args.directory
    if not path:
        path = load_experimental_data()

    edf_path = convert_to_edf(
        path,
        use_event_durations=False,
        write_targetness=True,
        overwrite=True,
        pre_filter=True)

    bdf_path = convert_to_bdf(
        path,
        use_event_durations=False,
        write_targetness=True,
        overwrite=True,
        pre_filter=True)

    if args.plot:
        plot_edf(edf_path)  # comment if not in an iPython notebook to plot using MNE

    print(f"\nWrote edf file to {edf_path}")
    print(f"\nWrote bdf file to {bdf_path}")
