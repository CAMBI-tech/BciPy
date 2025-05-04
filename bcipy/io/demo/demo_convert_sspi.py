"""Demonstrates converting BciPy data output to other EEG formats.

To use at bcipy root,

    `python bcipy/io/demo/demo_convert.py -d "path://to/bcipy/data/folder"`
"""
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import mne_bids as biddy
from tqdm import tqdm

from bcipy.io.convert import ConvertFormat, convert_to_bids_drowsiness, convert_eyetracking_to_bids, \
    convert_to_bids
from bcipy.io.load import BciPySessionTaskData, load_bcipy_data
from bcipy.io.utils import extract_task_type, extract_session_type, sort_by_timestamp

EXCLUDED_TASKS = ['Report', 'Offline', 'Intertask', 'BAD']


def load_historical_bcipy_data(directory: str, experiment_id: str,
                               task_name: str = 'Calibration') -> List[BciPySessionTaskData]:
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

        for ses_id, ses_path in enumerate(participant.iterdir()):
            if not ses_path.is_dir():
                continue

            tasks = [task_run for task_run in ses_path.iterdir() if task_run.is_dir()]
            sorted_tasks = sort_by_timestamp(tasks)

            for i, task_run in enumerate(sorted_tasks):
                task_type = extract_session_type(task_run)

                session_data = BciPySessionTaskData(
                    path=task_run,
                    user_id=user_id,
                    experiment_id=experiment_id,
                    session_id=ses_id,
                    run=i,
                    task_name=task_type
                )
                experiment_data.append(session_data)

    sorted_experiment_data = sorted(experiment_data, key=lambda x: str(x.path))
    return sorted_experiment_data


def convert_experiment_to_bids(
        directory: str,
        experiment_id: str,
        format: ConvertFormat = ConvertFormat.BV,
        output_dir: Optional[str] = None,
        include_eye_tracker: bool = True
) -> Path:
    """Converts the data in the study folder to BIDS format."""

    experiment_data = load_historical_bcipy_data(
        directory,
        experiment_id=""
    )

    # # Use for data post-2.0rc4
    # experiment_data = load_bcipy_data(directory, experiment_id, excluded_tasks=EXCLUDED_TASKS)

    print(f"Found {len(experiment_data)} sessions in folder ...")

    if not output_dir:
        output_dir = directory

    errors = []
    log_file_path = Path(output_dir) / f"{experiment_id}_conversion_log.log"
    with open(log_file_path, "w") as log_file:

        for data in tqdm(experiment_data, total=len(experiment_data), desc="Converting to BIDS",
                         unit="dir"):
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
                    convert_eyetracking_to_bids(
                        raw_data_path=data.path,
                        participant_id=data.user_id,
                        session_id=data.session_id,
                        run_id=data.run,
                        output_dir=str(bids_path),
                        task_name=data.task_name
                    )
            except Exception as e:
                error_msg = f"Error {data.path}:\n{str(e)}\n\n"
                log_file.write(error_msg)
                errors.append(data.path)

        print("\n-------------------- Conversion Summary --------------------\n")
        if errors:
            print(f"{len(errors)} session(s) failed. Details saved in: {log_file_path}")
        else:
            print("All sessions converted successfully!")

    bids_output = Path(f'{output_dir}/bids_{experiment_id}/')
    print(f"Data converted to BIDS format in: {bids_output}/")
    print("--------------------")

    return bids_output


if __name__ == "__main__":
    path = "/Users/srikarananthoju/cambi/data/SSPI_Main"
    salt = str(uuid.uuid4())[:4]
    experiment_id = "sspi_experiment_" + salt
    # convert a study to BIDS format
    bids_path_root = convert_experiment_to_bids(
        path,
        experiment_id,
        ConvertFormat.BV,
        output_dir="/Users/srikarananthoju/cambi/BciPy/bids_generated",
    )

    # --- Post Processing --- #

    # Updating dataset description file
    description_params = {
        "path": bids_path_root,
        "name": "CAMBI_Drowsiness_Experiment",
        "dataset_type": "raw",
        "data_license": None,
        "authors": "Daniel Klee, Betts Peters, Michelle Kinsela, Tab Memmott, Srikar,  MFO",
        "how_to_acknowledge": None,
        "funding": None,
        "references_and_links": "",
        "overwrite": True
    }
    biddy.make_dataset_description(**description_params)
