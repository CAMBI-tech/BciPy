from bcipy.acquisition.devices import DeviceSpec
from bcipy.helpers.validate import validate_experiments
import errno
import os
from time import localtime, strftime
from shutil import copyfile
from pathlib import Path
import json

from bcipy.config import DEFAULT_ENCODING, DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETER_FILENAME


def save_json_data(data: dict, location: str, name: str) -> str:
    """
    Writes Parameters as a json file.

        parameters[dict]: dict of configuration
        location[str]: directory in which to save
        name[str]: optional name of file; default is parameters.json

    Returns path of saved file
    """
    path = Path(location, name)
    with open(Path(location, name), 'w', encoding=DEFAULT_ENCODING) as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)
    return str(path)


def save_experiment_data(experiments, fields, location, name) -> str:
    validate_experiments(experiments, fields)
    return save_json_data(experiments, location, name)


def save_field_data(fields, location, name) -> str:
    return save_json_data(fields, location, name)


def save_experiment_field_data(data, location, name) -> str:
    return save_json_data(data, location, name)


def init_save_data_structure(data_save_path: str,
                             user_id: str,
                             parameters: str,
                             task: str,
                             experiment_id: str = DEFAULT_EXPERIMENT_ID
                             ) -> str:
    """
    Initialize Save Data Structure.

        data_save_path[str]: string of path to save our data in
        user_id[str]: string of user name / related information
        parameters[str]: parameter location for the experiment
        experiment_id[str]: Name of the experiment. Default name is DEFAULT_EXPERIMENT_ID.
        task[str]: name of task type. Ex. RSVP Calibration

    """

    # make an experiment folder : note datetime is in utc
    save_folder_name = f'{data_save_path}{experiment_id}/{user_id}'
    dt = strftime('%a_%d_%b_%Y_%Hhr%Mmin%Ssec_%z', localtime())
    task = task.replace(' ', '_')
    save_directory = f"{save_folder_name}/{user_id}_{task}_{dt}"

    try:
        # make a directory to save data to
        os.makedirs(save_folder_name)
        os.makedirs(save_directory)
        os.makedirs(os.path.join(save_directory, 'logs'), exist_ok=True)

    except OSError as error:
        # If the error is anything other than file existing, raise an error
        if error.errno != errno.EEXIST:
            raise error

        # since this is only called on init, we can make another folder run
        os.makedirs(save_directory)
        os.makedirs(os.path.join(save_directory, 'logs'), exist_ok=True)

    copyfile(parameters, Path(save_directory, DEFAULT_PARAMETER_FILENAME))

    return save_directory


def save_device_spec(device_spec: DeviceSpec, location: str, name: str) -> str:
    """
    Save device spec to a json file.

    Parameters:
        device_spec (DeviceSpec): device spec to save
        location (str): location to save the file
        name (str): name of the file to be saved

    Returns:
        str: path to the saved file
    """
    return save_json_data(device_spec.to_dict(), location, name)


def _save_session_related_data(file, session_dictionary):
    """
    Save Session Related Data.
    Parameters
    ----------
        file[str]: string of path to save our data in
        session_dictionary[dict]: dictionary of session data. It will appear
            as follows:
                {{ "series": {
                        "1": {
                          "0": {
                            "target_text": "COPY_PHRASE",
                            "current_text": "COPY_",
                            "eeg_len": 22,
                            "next_display_state": "COPY_",
                            "stimuli": [["+", "_", "G", "L", "B"]],
                            "target_info": [
                              "nontarget", ... ,
                            ],
                            "timing_sti": [[1, 0.2, 0.2, 0.2, 0.2]],
                            "triggers": [[ "+", 0.0], ["_", 0.9922] ..],
                            },
                        ... ,
                        "7": {
                            ... ,
                  },
                  "mode": "RSVP",
                  "session": "data/demo_user/demo_user",
                  "task": "Copy Phrase",
                  "total_time_spent": 83.24798703193665
                }}
    Returns
    -------
        file, session data file (json file)
    """
    # Try opening as json, if not able to use open() to create first
    try:
        file = json.load(file, 'wt')
    except BaseException:
        file = open(file, 'wt', encoding=DEFAULT_ENCODING)

    # Use the file to dump data to
    json.dump(session_dictionary, file, indent=2)
    return file
