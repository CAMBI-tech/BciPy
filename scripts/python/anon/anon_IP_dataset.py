from bcipy.io.load import load_experimental_data
from pathlib import Path
import random


import json
import shutil
from pathlib import Path

from bcipy.core.parameters import DEFAULT_PARAMETERS_PATH, Parameters

def update(params_path: str) -> None:
    """Update the parameters at the given path"""
    default_params = Parameters(DEFAULT_PARAMETERS_PATH, cast_values=True)

    # open as json, not Parameters, to avoid validation checks
    with open(params_path, 'r', encoding='utf8') as json_file:
        params = json.load(json_file)

    # rename attributes if needed
    mapping = {"readableName": "name", "recommended_values": "recommended"}

    updated_params = {}
    for key, entry in params.items():
        editable = False
        if key in default_params:
            editable = default_params.get_entry(key)['editable']
        value = {mapping.get(k, k): v for k, v in entry.items()}

        if "editable" not in value:
            value["editable"] = editable
        updated_params[key] = value

    # overwrite json file if needed
    original_path = Path(params_path)
    shutil.copyfile(original_path,
                    Path(original_path.parent, "parameters_original.json"))

    with open(params_path, 'w', encoding='utf8') as json_path:
        json.dump(updated_params, json_path, ensure_ascii=False, indent=2)

    # load from overwritten file
    parameters = Parameters(source=params_path, cast_values=True)

    added_params = [
        key for key, change in default_params.diff(parameters).items()
        if change.original_value is None
    ]

    parameters.save()


def gen_IDs(length_to_generate):
    # Generate a list of unique IDs without repeating
    ID_list = []
    for i in range(length_to_generate):
        ID_list.append(f"00{i+5}")
    random.shuffle(ID_list)
    return ID_list


def update_labels_and_remove_logs(part_path, new_ID):
    # Remove the labels, datetime and logs from the dataset
    invalid_task = []
    for task_path in part_path.iterdir():
        if task_path.is_dir():
            # pull the task type from the path
            task_mode = task_path.name.split('_')[1]
            task_type = task_path.name.split('_')[2]
            timestamp = task_path.name.split('_')[-2]
            condition = ''
            if task_type == 'Copy':
                task_type = 'CopyPhrase'
            
            # check for experimental data
            if not (task_path / 'raw_data.csv').exists():
                print(f"Experiment data not found in {task_path}. Skipping...")
                invalid_task.append(task_path)
                continue
            # open the experimental data path
            with open(task_path / 'experiment_data.json', 'r', encoding='utf8') as json_file:
                experiment_data = json.load(json_file)
                experiment_condition = experiment_data['Condition']
                condition = experiment_condition

            if task_type == 'CopyPhrase':
                # open the session data and anonymize it
                with open(task_path / 'session.json', 'r', encoding='utf8') as json_file:
                    # load the session data and anonymize it
                    session_data = json.load(json_file)
                    session_data['session'] = 'ANONYMIZED'
                
                # delete the session data file
                session_path = task_path / 'session.json'
                session_path.unlink()
                # save the session data to a new file
                with open(task_path / 'session.json', 'w', encoding='utf8') as json_file:
                    # save the session data to a new file
                    json.dump(session_data, json_file, ensure_ascii=False, indent=2)            
           
            for file in task_path.iterdir():
                if file.is_dir():
                    # Remove the logs and labels
                    if file.name == 'logs':
                        shutil.rmtree(file)
                elif file.name == 'parameters.json':
                    update(file)
                elif ".xlsx" in file.name:
                    # delete the xlsx files
                    file.unlink()
                elif ".pkl" in file.name:
                    # delete the pkl files
                    file.unlink()
        # update the task name
        task_path.rename(task_path.parent / f"{new_ID}_{task_mode}_{task_type}_{condition}_{timestamp}")
    
    # update the part name
    part_path.rename(part_path.parent / f"{new_ID}")

    return invalid_task


def get_dir_count(path):
    # Get the number of directories in the path
    count = 0
    for item in path.iterdir():
        if item.is_dir():
            count += 1
    return count

if __name__ == '__main__':
    # Load the experimental data
    data_path = Path(load_experimental_data())

    ids_to_generate = get_dir_count(data_path)
    # Generate the IDs
    new_IDs = gen_IDs(ids_to_generate)

    ID_map = []
    invalid_tasks = []
    for part, ID in zip(data_path.iterdir(), new_IDs):
        ID_map.append((ID, part.name))
        invalid_task = update_labels_and_remove_logs(part, ID)
        if invalid_task:
            invalid_tasks.append(invalid_task)
            print(f"Invalid task found in {part.name}: {invalid_task}")

        print(f"Updated {part.name} to {ID}")

    # Print the invalid tasks  
    print("Invalid tasks:")
    for task in invalid_tasks:
        print(task)
    
    print("ID map:")
    for ID, part in ID_map:
        print(f"{ID}: {part}")
    breakpoint()
    # Save the ID map
    with open('ID_map.txt', 'w') as file:
        for ID, part in ID_map:
            file.write(f"{ID}: {part}\n")
