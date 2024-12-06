"""Demonstrates converting BciPy data output to other EEG formats.

To use at bcipy root,

    `python bcipy/io/demo/demo_convert.py -d "path://to/bcipy/data/folder"`
"""
from typing import Optional, List
from pathlib import Path
from bcipy.io.convert import convert_to_bids, ConvertFormat, convert_eyetracking_to_bids
from bcipy.gui.file_dialog import ask_directory
from bcipy.io.load import BciPySessionTaskData

EXCLUDED_TASKS = ['Report', 'Offline', 'Intertask', 'BAD']
# Note: We can share the eye tracking data as-is but we need 
# sub1/eeg/.tsv sub1/eyetracker/.tsv

def load_bcipy_data(directory: str, experiment_id: str) -> List[BciPySessionTaskData]:
    """Load the data from the given directory.
    
    The expected directory structure is:

    directory/
        user1/
            task_run/
                raw_data.csv
        user2/
            task_run/
                raw_data.csv
    """

    data = Path(directory)
    experiment_data = []
    for participant in data.iterdir():
        
        # Skip files
        if not participant.is_dir():
            continue

        # pull out the user id. This is the name of the folder
        user_id = participant.name

        for task_run in participant.iterdir():
            if not task_run.is_dir():
                continue

            task_name = "MatrixCalibration"
            session_data = BciPySessionTaskData(
                path=task_run,
                user_id=user_id,
                experiment_id=experiment_id,
                session_id=1,
                run=1,
                task_name=task_name
            )
            experiment_data.append(session_data)
    return experiment_data


def convert_experiment_to_bids(
        directory: str,
        experiment_id: str,
        format: ConvertFormat = ConvertFormat.BV,
        output_dir: Optional[str] = None,
        include_eye_tracker: bool = False) -> None:
    """Converts the data in the study folder to BIDS format."""

    experiment_data = load_bcipy_data(
        directory,
        experiment_id=experiment_id)

    if not output_dir:
        output_dir = directory

    errors = []
    for data in experiment_data:
        try:
            bids_path = convert_to_bids(
                data_dir=data.path,
                participant_id=data.user_id,
                session_id=data.session_id,
                run_id=data.run,
                task_name=data.task_name,
                output_dir=f'{output_dir}/bids_{experiment_id}/',
                format=format
            )

            if include_eye_tracker:
                # Convert the eye tracker data
                # The eye tracker data is in the same folder as the EEG data, but should be saved in a different folder
                bids_path = Path(bids_path).parent
                convert_eyetracking_to_bids(
                    raw_data_path=data.path,
                    participant_id=data.user_id,
                    session_id=data.session_id,
                    run_id=data.run,
                    output_dir=bids_path,
                    task_name=data.task_name
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
        default='Matrix Multimodal Experiment',
    )

    args = parser.parse_args()

    path = args.directory
    if not path:
        path = ask_directory("Select the directory with data to be converted")

    # convert a study to BIDS format
    convert_experiment_to_bids(path, args.experiment, ConvertFormat.BV, output_dir=".", include_eye_tracker=True)
