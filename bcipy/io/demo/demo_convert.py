"""Demonstrates converting BciPy data output to other EEG formats.

To use at bcipy root,

    `python bcipy/io/demo/demo_convert.py -d "path://to/bcipy/data/folder"`
"""
from typing import Optional
from bcipy.io.convert import convert_to_bids, ConvertFormat
from bcipy.gui.file_dialog import ask_directory
from bcipy.io.load import load_bcipy_data

EXCLUDED_TASKS = ['Report', 'Offline', 'Intertask', 'BAD']


def convert_experiment_to_bids(
        directory: str,
        experiment_id: str,
        format: ConvertFormat = ConvertFormat.BV,
        output_dir: Optional[str] = None) -> None:
    """Converts the data in the study folder to BIDS format."""

    experiment_data = load_bcipy_data(
        directory,
        experiment_id=experiment_id,
        excluded_tasks=EXCLUDED_TASKS,
        anonymize=False)
    if not output_dir:
        output_dir = directory

    errors = []
    for data in experiment_data:
        try:
            convert_to_bids(
                data_dir=data.path,
                participant_id=data.user_id,
                session_id=data.session_id,
                run_id=data.run,
                task_name=data.task_name,
                output_dir=f'{output_dir}/bids_{experiment_id}/',
                format=format
            )
        except Exception as e:
            print(f"Error converting {data.path} - {e}")
            errors.append(str(data.path))

    print("--------------------")
    if errors:
        print(f"Errors converting the following data: {errors}")

    print(f"\nData converted to BIDS format in {output_dir}/bids_{experiment_id}/")
    print("--------------------")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--directory',
        help='Path to the directory with raw_data to be converted',
        required=False)
    parser.add_argument(
        '-e',
        '--experiment',
        help='Experiment ID to convert',
        default='SCRD',
    )

    args = parser.parse_args()

    path = args.directory
    if not path:
        path = ask_directory("Select the directory with data to be converted")

    # convert a study to BIDS format
    convert_experiment_to_bids(path, args.experiment, ConvertFormat.BV)
