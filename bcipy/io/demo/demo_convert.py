"""Demonstrates converting BciPy data output to other EEG formats.

To use at bcipy root,

    `python bcipy/io/demo/demo_convert.py -d "path://to/bcipy/data/folder"`
"""
from typing import Optional, List, OrderedDict
from pathlib import Path
from bcipy.io.convert import convert_to_bids, ConvertFormat, convert_eyetracking_to_bids
from bcipy.gui.file_dialog import ask_directory
from bcipy.io.load import BciPySessionTaskData, load_bcipy_data
import mne_bids as biddy
import argparse

EXCLUDED_TASKS = ['Report', 'Offline', 'Intertask', 'BAD']


def load_historical_bcipy_data(directory: str, experiment_id: str,
                               task_name: str = 'MatrixCalibration') -> List[BciPySessionTaskData]:
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

        for i, task_run in enumerate(participant.iterdir()):
            if not task_run.is_dir():
                continue

            session_data = BciPySessionTaskData(
                path=task_run,
                user_id=user_id,
                experiment_id=experiment_id,
                session_id=i,
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
        include_eye_tracker: bool = False
) -> Path:
    """Converts the data in the study folder to BIDS format."""

    # Use for data pre-2.0rc4
    experiment_data = load_historical_bcipy_data(
        directory,
        experiment_id="")

    # Use for data post-2.0rc4
    # experiment_data = load_bcipy_data(directory, experiment_id, excluded_tasks=EXCLUDED_TASKS)

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
                format=format,
            )

            if include_eye_tracker:
                # Convert the eye tracker data
                # The eye tracker data is in the same folder as the EEG data, but should be saved in a different folder
                bids_path = Path(bids_path).parent
                try:
                    convert_eyetracking_to_bids(
                        raw_data_path=data.path,
                        participant_id=data.user_id,
                        session_id=data.session_id,
                        run_id=data.run,
                        output_dir=bids_path,
                        task_name=data.task_name
                    )
                except Exception as e:
                    print(f"Error converting eye tracker data for {data.path} - {e}")
                    errors.append(f"Error converting eye tracker data for {data.path}")

        except Exception as e:
            print(f"\nError converting {data.path} : \n{e}\n")
            errors.append(str(data.path))

    print("--------------------")
    if errors:
        print(f"{len(errors)} Errors converting ==> {errors}")

    bids_output = Path(f'{output_dir}/bids_{experiment_id}/')
    print(f"\nData converted to BIDS format in {bids_output}/")
    print("--------------------")
    return bids_output


if __name__ == '__main__':
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
    parser.add_argument(
        '-et',
        '--eye_tracker',
        help='Include eye tracker data',
        default=False,
        action='store_true',
    )

    args = parser.parse_args()

    path = args.directory
    if not path:
        path = ask_directory("Select the directory with data to be converted", strict=True)

    # convert a study to BIDS format
    bids_path_root = convert_experiment_to_bids(
        path,
        args.experiment,
        ConvertFormat.EDF,
        output_dir="/Users/srikarananthoju/cambi/BciPy",
        include_eye_tracker=args.eye_tracker
    )

    # --- Post Processing --- #

    # Updating dataset description file
    description_params = {
        "path": bids_path_root,
        "name": "CAMBI_MultiModal_Experiment",
        "dataset_type": "raw",
        "data_license": None,
        "authors": None,
        "how_to_acknowledge": None,
        "funding": None,
        "references_and_links": None,
        "overwrite": True
    }
    biddy.make_dataset_description(**description_params)
