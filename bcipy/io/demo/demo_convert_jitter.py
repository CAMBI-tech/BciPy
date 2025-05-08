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
from bcipy.io.utils import extract_task_type, extract_session_type, sort_by_timestamp, \
    extract_timestamp_jitter, extract_jitter_type

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

        task_runs = [task_run for task_run in participant.iterdir() if task_run.is_dir()]
        task_runs = sorted(task_runs, key=lambda x: extract_timestamp_jitter(x))
        jitter_types: list = [extract_jitter_type(task_run) for task_run in task_runs]
        jitter_types[-1] = "-1"
        jitter_map = {"0.05": "JitterShort", "0.1": "JitterLong", "0.0": "NoJitter", "-1": "NoJitterSlow"}
        jitter_types = [jitter_map.get(jitter_type) for jitter_type in jitter_types]
        jitter_type_ordering = []
        for i, jitter_type in enumerate(jitter_types):
            if jitter_type not in jitter_type_ordering:
                jitter_type_ordering.append(jitter_type)
        for i, task_run in enumerate(task_runs):
            task_name = extract_session_type(str(task_run))
            jitter_type = jitter_types[i]
            session_index = jitter_type_ordering.index(jitter_type)
            session_data = BciPySessionTaskData(
                path=task_run,
                user_id=user_id,
                experiment_id=experiment_id,
                session_id=f"{session_index}XX{jitter_type}",
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
        include_eye_tracker: bool = True
) -> Path:
    """Converts the data in the study folder to BIDS format."""

    experiment_data = load_historical_bcipy_data(
        directory,
        experiment_id=""
    )

    # # Use for data post-2.0rc4
    # experiment_data = load_bcipy_data(directory, experiment_id, excluded_tasks=EXCLUDED_TASKS)

    print(f"Found {len(experiment_data)} sessions in folder {directory}")
    if not experiment_data:
        print("No data found. Exiting.")
        exit(1)

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
    path = "/Users/srikarananthoju/cambi/data/jitter"
    salt = str(uuid.uuid4())[:4]
    experiment_id = "jitter_experiment_" + salt
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
        "name": "CAMBI_Jitter_Experiment",
        "dataset_type": "raw",
        "data_license": None,
        "authors": "Daniel Klee, Betts Peters, Michelle Kinsela, Tab Memmott, Srikar,  MFO",
        "how_to_acknowledge": None,
        "funding": None,
        "references_and_links": "",
        "overwrite": True
    }
    biddy.make_dataset_description(**description_params)
