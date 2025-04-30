"""Demonstrates converting BciPy data output to other EEG formats.

To use at bcipy root,

    `python bcipy/io/demo/demo_convert.py -d "path://to/bcipy/data/folder"`
"""
from operator import index
from typing import Optional, List, OrderedDict
from pathlib import Path

from tqdm import tqdm

from bcipy.io.convert import convert_to_bids, ConvertFormat, convert_eyetracking_to_bids
from bcipy.gui.file_dialog import ask_directory
from bcipy.io.load import BciPySessionTaskData, load_bcipy_data
import mne_bids as biddy
import argparse

from bcipy.io.utils import extract_task_type, extract_timestamp

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

        # To group task runs by task type
        for i, task_run in enumerate(participant.iterdir()):
            if not task_run.is_dir():
                continue

            task_type = extract_task_type(str(task_run), default=task_name)

            session_data = BciPySessionTaskData(
                path=task_run,
                user_id=user_id,
                experiment_id=experiment_id,
                session_id=i,
                run=1,
                task_name=task_type
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

    task_group_order = ['CALIBRATION', 'IPC', 'IPS', 'IPO', 'NP']
    task_group_order_map = {task: i for i, task in enumerate(task_group_order)}
    # Organizing task runs by session type (IPO, IPS, NP, IPC,CALIBRATION)
    task_groups = {}
    for data in experiment_data:
        task_group_key = (data.user_id, data.task_name)
        if task_group_key in task_groups:
            task_groups[task_group_key].append(data.path)
        else:
            task_groups[task_group_key] = [data.path]

    breakpoint()
    subject_task_timestamps = {}
    for key, val in task_groups.items():
        sorted_paths = sorted(val)
        task_groups[key] = sorted_paths # sorting the task runs
        # Needed for finding task run ordering
        sub, task_name = key
        earliest_task_path = sorted_paths[0]
        dt = extract_timestamp(str(earliest_task_path))
        if sub not in subject_task_timestamps:
            subject_task_timestamps[sub] = []
        subject_task_timestamps[sub].append((task_name, dt))

    subject_task_orderings = {}
    for key, val in subject_task_timestamps.items():
        # Sort the task runs by timestamp
        sorted_tasks = sorted(val, key=lambda x: x[1])
        subject_task_orderings[key] = [x[0] for x in sorted_tasks]

    for data in tqdm(experiment_data, desc="Processing data", unit="dir"):
        try:
            task_group_key = (data.user_id, data.task_name)
            task_group_value = task_groups[task_group_key]
            sub = data.user_id
            run_id = task_group_value.index(data.path) # picking the run id from the sorted task runs
            session_id = subject_task_orderings[sub].index(data.task_name) # picking session_id from sorted task types
            bids_path = convert_to_bids(
                data_dir=data.path,
                participant_id=data.user_id,
                session_id=str(session_id),
                run_id=str(run_id),
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

if __name__ == "__main__":
    print(
        "This script is intended to be run directly. Use the command line interface to execute it.")
    path = "/Users/srikarananthoju/cambi/data/inquirypreview"
    experiment_id = "inquiry_preview_corrected_order"
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
        "name": "CAMBI_Inquiry_Preview_Experiment",
        "dataset_type": "raw",
        "data_license": None,
        "authors": "Daniel Klee, Betts Peters, Michelle Kinsela, Tab Memmott, Srikar,  MFO",
        "how_to_acknowledge": None,
        "funding": None,
        "references_and_links": (
            "Peters B, Celik B, Gaines D, Galvin-McLaughlin D, Imbiriba T, Kinsella M, "
            "Klee D, Lawhead M, Memmott T, Smedemark-Margulies N, Wiedrick J, Erdogmus D, "
            "Oken B, Vertanen K, Fried-Oken M. RSVP keyboard with inquiry preview: mixed "
            "performance and user experience with an adaptive, multimodal typing interface "
            "combining EEG and switch input. J Neural Eng. 2025 Feb 4;22(1). doi: "
            "10.1088/1741-2552/ada8e0. PMID: 39793200."
        ),
        "overwrite": True
    }
    biddy.make_dataset_description(**description_params)
