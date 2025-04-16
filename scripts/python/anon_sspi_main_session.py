from bcipy.io.load import load_experimental_data, load_json_parameters
from pathlib import Path
import random
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

    for participant_run in data_path.iterdir():
        if participant_run.is_dir():
            # Get the participant ID from the folder name
            file_name_parts = participant_run.name.split('_')
            participant_id = file_name_parts[0]
            task_type = file_name_parts[1]
            task_name = file_name_parts[2]
            day = file_name_parts[3]
            date = file_name_parts[4]
            time = file_name_parts[5]
            if task_type == 'Copy':
                task_name = 'CopyPhrase'
                day = file_name_parts[4]


            



if __name__ == "__main__":
    # Load the experimental data
    path_to_bids = Path(load_experimental_data())
    print(path_to_bids)

    # Load the parameters from the default path
    params = load_json_parameters(DEFAULT_PARAMETERS_PATH)
    partipant_count = len(path_to_bids.iterdir())
    print(partipant_count)
    # Update the parameters
    # update()

    # Generate unique IDs
    ID_list = gen_IDs(10)
    print(ID_list)