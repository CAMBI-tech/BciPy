from bcipy.io.load import load_experimental_data, load_json_parameters
from pathlib import Path
import random
from datetime import datetime
import shutil
import json
import os
from bcipy.core.parameters import DEFAULT_PARAMETERS_PATH, Parameters

def gen_IDs(length_to_generate):
    # Generate a list of unique IDs without repeating
    ID_list = []
    for i in range(length_to_generate):
        ID_list.append(f"00{i+5}")
    random.shuffle(ID_list)
    return ID_list


def anon_data(data_path, new_participant_id):
    """Load the experimental data from the given path and anonymize it.
    
    Each datapath is expected to contain a participant folder with the data files.

    The data files are in BciPy format and consist of two days of data where each day contains
    a calibration for matrix and rsvp and two copy phrase tasks. The order of matrix or rsvp is random.

    The data files are expected to be in the following format:
    - <participant_id>_<task>_<day>_<date>_<time>/raw_data.csv
    - <participant_id>_<task>_<day>_<date>_<time>/parameters.json
    - <participant_id>_<task>_<day>_<date>_<time>/devices.json
    - <participant_id>_<task>_<day>_<date>_<time>/triggers.txt
    - <participant_id>_<task>_<day>_<date>_<time>/session.json

    We want to strip the participant_id from the file names and replace it with a unique ID. As well, remove the information from the session.json file.
    """

    run_days = {}
    for participant_run in data_path.iterdir():
        if participant_run.is_dir():

            # # Get the participant ID from the folder name
            file_name_parts = participant_run.name.split('_')
            participant_id = file_name_parts[0]
            if participant_id not in run_days:
                run_days[participant_id] = {}
            task_type = file_name_parts[1]
            task_name = file_name_parts[2]
            day = file_name_parts[3]
            date_int = file_name_parts[4]
            month = file_name_parts[5]
            year = file_name_parts[6]
            time = file_name_parts[7]
            offset = file_name_parts[-1]
            if task_name == 'Copy':
                task_name = 'CopyPhrase'
                day = file_name_parts[4]
                date_int = file_name_parts[5]
                month = file_name_parts[6]
                year = file_name_parts[7]
                time = file_name_parts[8]

            # breakpoint()
            tm_hr = time.split('hr')[0]
            tm_min = time.split('hr')[1].split('min')[0]
            tm_sec = time.split('hr')[1].split('min')[1].split('sec')[0]
            date_str = f"{year}-{month}-{date_int} {tm_hr}:{tm_min}:{tm_sec}"
            datetime_str = datetime.strptime(date_str, '%Y-%b-%d %H:%M:%S')
            
            date = datetime_str.strftime('%Y-%m-%d')
            if date not in run_days[participant_id]:
                run_days[participant_id][date] = []
 
            run_data = {}
            run_data['task'] = task_name
            run_data['path'] = participant_run
            run_data['newID'] = new_participant_id
            run_data['oldID'] = participant_id
            run_data['datetime'] = datetime_str
            run_data['newFilePath'] = f'{new_participant_id}_{task_type}_{task_name}_{int(datetime_str.timestamp())}'

            run_days[participant_id][date].append(run_data)
            

            # Create the new file name with the unique ID
    return run_days

            
def add_run_id_to_data(user_data):

    new_user_data = []
    for run_days in user_data:
        new_run_days = {}
        for participant_id, days in run_days.items():
            # there are two days of data for each participant, find which one went first and assign it an ID of 1 and the other an ID of 2
            # There should only be two days of data for each participant, assert that this is the case
            if len(days) == 2:
                # using the date of %Y-%m-%d, sort the days
                day_one = sorted(days.keys())[0]
                day_two = sorted(days.keys())[1]

                # assign the first day an ID of 1 and the second day an ID of 2
                new_run_days[participant_id] = {}
                new_run_days[participant_id][1] = days[day_one]
                new_run_days[participant_id][2] = days[day_two]
            elif len(days) > 2:
                raise ValueError(f"More than two days of data for participant {participant_id}.")
            else:
                new_run_days[participant_id] = {}
                new_run_days[participant_id][1] = days[sorted(days.keys())[0]]
        
        new_user_data.append(new_run_days)

    return new_user_data


def anon_run_id_data(user_data, new_data_path):

    for user in user_data:
        for participant_id, run_id_data in user.items():
            # make a new directory for the participant
            new_id = run_id_data[1][0]['newID']
            new_participant_path = new_data_path / new_id
            new_participant_path.mkdir(parents=True, exist_ok=True)
            for run_id, run_data in run_id_data.items():

                # make a new directory for the run ID
                new_run_id_path = new_participant_path / str(run_id)
                new_run_id_path.mkdir(parents=True, exist_ok=True)

                # Get the new path to save the data
                for runs in run_data:
                    if isinstance(runs, dict):
                        # Get the new file path
                        new_file_path = new_run_id_path / runs['newFilePath']
                        
                        new_file_path.mkdir(parents=True, exist_ok=False)
                        # Copy the data to the new path

                        data_path = runs['path']

                        for file in data_path.iterdir():
                            if file.is_file():
                                # Copy the file to the new path
                                new_file = new_file_path / file.name
                                # Copy the file to the new path
                                if file.name == 'triggers.txt' or '.csv' in file.name or file.name == 'parameters.json' or file.name == 'devices.json':
                                    shutil.copy(file, new_file)
                                    print(f"Copied {file} to {new_file}")
                                
                                if file.name == 'session.json':
                                    # Load the session.json file and remove the participant ID
                                    with open(file, 'r') as f:
                                        session_data = json.load(f)
                                        session_data['session'] = "ANONYMIZED"
                                    
                                    # Save the new session.json file
                                    with open(new_file, 'w') as f:
                                        json.dump(session_data, f)
                                        print(f"Saved {new_file}")
                    else:
                        breakpoint()
                # run_data['path'].rename(new_file_path)
                # print(f"Renamed {run_data['path']} to {new_file_path}")

                # break_


if __name__ == "__main__":
    # Load the experimental data
    path_to_bids = Path(load_experimental_data())
    print(path_to_bids)

    # Load the parameters from the default path
    # params = load_json_parameters(DEFAULT_PARAMETERS_PATH)
    # Update the parameters
    # update()

    # Generate unique IDs
    ID_list = gen_IDs(10)
    print(ID_list)

    # new path
    new_path_to_data = './anon_data/'
    new_path_to_data = Path(new_path_to_data)
    new_path_to_data.mkdir(parents=True, exist_ok=True)

    date_filtered_data = []
    for i, participant_run in enumerate(path_to_bids.iterdir()):
        if participant_run.is_dir():
            # Get the participant ID from the folder name
            participant_id = participant_run.name
            new_participant_id = ID_list[i]
            print(f"Anonymizing {participant_id} to {new_participant_id}")

            # Anonymize the data
            run_days = anon_data(participant_run, new_participant_id)
            date_filtered_data.append(run_days)

    run_id_data = add_run_id_to_data(date_filtered_data)

    # export the run_id to oldID mapping to a json file
    run_id_mapping = {}
    for data_run in run_id_data:
        for participant_id, run_id in data_run.items():
            for run_id, run_data in run_id.items():
                run_id_mapping[participant_id] = run_data[0]['newID']
    run_id_mapping_path = new_path_to_data / 'sspi_main_anon_map.json'
    with open(run_id_mapping_path, 'w') as f:
        json.dump(run_id_mapping, f)
        print(f"Saved {run_id_mapping_path}")

    
    # breakpoint()
    response = anon_run_id_data(run_id_data, new_path_to_data)
    breakpoint()
