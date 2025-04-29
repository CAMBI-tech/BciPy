from bcipy.io.load import load_experimental_data
from pathlib import Path
import random


import json
import shutil
from pathlib import Path

from bcipy.core.parameters import DEFAULT_PARAMETERS_PATH, Parameters

def anonymize_data(part_path: str, new_ID) -> None:
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
                elif ".xlsx" in file.name:
                    # delete the xlsx files
                    file.unlink()
                elif ".pkl" in file.name:
                    # delete the pkl files
                    file.unlink()
                elif ".png" in file.name:
                    # delete the png files
                    file.unlink()
                elif ".bdf" in file.name or ".edf" in file.name:
                    # delete the bdf and edf files
                    file.unlink()
        # update the task name
        task_path.rename(task_path.parent / f"{new_ID}_{task_mode}_{task_type}_{timestamp}")
    
    # update the part name
    part_path.rename(part_path.parent / f"{new_ID}")
    return invalid_task

def gen_IDs(length_to_generate):
    # Generate a list of unique IDs without repeating
    ID_list = []
    for i in range(length_to_generate):
        ID_list.append(f"00{i+5}")
    random.shuffle(ID_list)
    return ID_list

if __name__ == "__main__":
    # Example usage
    data_path = Path(load_experimental_data())
    
    # get the number of IDS to generate based on the number of directories in the data path
    length_to_generate = len([d for d in data_path.iterdir() if d.is_dir()])
    generated_IDs = gen_IDs(length_to_generate)
    print(generated_IDs)  # Print the generated IDs

    id_map = []
    for i, folder in enumerate(data_path.iterdir()):
        if folder.is_dir():
            # Rename the folder to the generated ID
            new_id = f"{generated_IDs[i]}"
            old_id = folder.name
            id_map.append((old_id, new_id))
            response = anonymize_data(folder, new_id)
    
    breakpoint()
    # Print the ID map
    print("ID map:")
    for old_id, new_id in id_map:
        print(f"{old_id} -> {new_id}")
    # save the ID map to a file
    with open("ID_map.json", "w") as f:
        json.dump(id_map, f, indent=4)
            