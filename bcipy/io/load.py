# mypy: disable-error-code="arg-type, union-attr"
import json
import logging
import os
import pickle
from pathlib import Path
from shutil import copyfile
from time import localtime, strftime
from typing import List, Optional, Union

from bcipy.config import (DEFAULT_ENCODING, DEFAULT_EXPERIMENT_PATH,
                          DEFAULT_FIELD_PATH, DEFAULT_PARAMETERS_PATH,
                          DEFAULT_PARAMETERS_FILENAME,
                          EXPERIMENT_FILENAME, FIELD_FILENAME,
                          SIGNAL_MODEL_FILE_SUFFIX, SESSION_LOG_FILENAME)
from bcipy.gui.file_dialog import ask_directory, ask_filename
from bcipy.exceptions import (BciPyCoreException,
                              InvalidExperimentException)
from bcipy.core.parameters import Parameters
from bcipy.core.raw_data import RawData
from bcipy.preferences import preferences
from bcipy.signal.model import SignalModel

log = logging.getLogger(SESSION_LOG_FILENAME)


def copy_parameters(path: str = DEFAULT_PARAMETERS_PATH,
                    destination: Optional[str] = None) -> str:
    """Creates a copy of the given configuration (parameters.json) to the
    given directory and returns the path.

    Parameters:
    -----------
        path: str - optional path of parameters file to copy; used default if not provided.
        destination: str - optional destination directory; default is the same
          directory as the default parameters.
    Returns:
    --------
        path to the new file.
    """
    default_dir = str(Path(DEFAULT_PARAMETERS_PATH).parent)

    destination = default_dir if destination is None else destination
    filename = strftime('parameters_%Y-%m-%d_%Hh%Mm%Ss.json', localtime())

    path = str(Path(destination, filename))
    copyfile(DEFAULT_PARAMETERS_PATH, path)
    return path


def load_experiments(path: str = f'{DEFAULT_EXPERIMENT_PATH}/{EXPERIMENT_FILENAME}') -> dict:
    """Load Experiments.

    PARAMETERS
    ----------
    :param: path: string path to the experiments file.

    Returns
    -------
        A dictionary of experiments, with the following format:
            { name: { fields : {name: '', required: bool, anonymize: bool}, summary: '' } }

    """
    with open(path, 'r', encoding=DEFAULT_ENCODING) as json_file:
        return json.load(json_file)


def extract_mode(bcipy_data_directory: str) -> str:
    """Extract Mode.

    This method extracts the task mode from a BciPy data save directory. This is important for
        trigger conversions and extracting targeteness.

    *note*: this is not compatible with older versions of BciPy (pre 1.5.0) where
        the tasks and modes were considered together using integers (1, 2, 3).

    PARAMETERS
    ----------
    :param: bcipy_data_directory: string path to the data directory
    """
    directory = bcipy_data_directory.lower()
    if 'calibration' in directory:
        return 'calibration'
    elif 'copy' in directory:
        return 'copy_phrase'
    raise BciPyCoreException(f'No valid mode could be extracted from [{directory}]')


def load_fields(path: str = f'{DEFAULT_FIELD_PATH}/{FIELD_FILENAME}') -> dict:
    """Load Fields.

    PARAMETERS
    ----------
    :param: path: string path to the fields file.

    Returns
    -------
        A dictionary of fields, with the following format:
            {
                "field_name": {
                    "help_text": "",
                    "type": ""
            }

    """
    with open(path, 'r', encoding=DEFAULT_ENCODING) as json_file:
        return json.load(json_file)


def load_experiment_fields(experiment: dict) -> list:
    """Load Experiment Fields.

    {
        'fields': [{}, {}],
        'summary': ''
    }

    Using the experiment dictionary, loop over the field keys and put them in a list.
    """
    if isinstance(experiment, dict):
        try:
            return [name for field in experiment['fields'] for name in field.keys()]
        except KeyError:
            raise InvalidExperimentException(
                'Experiment is not formatted correctly. It should be passed as a dictionary with the fields and'
                f' summary keys. Fields is a list of dictionaries. Summary is a string. \n experiment=[{experiment}]')
    raise TypeError('Unsupported experiment type. It should be passed as a dictionary with the fields and summary keys')


def load_json_parameters(path: str, value_cast: bool = False) -> Parameters:
    """Load JSON Parameters.

    Given a path to a json of parameters, convert to a dictionary and optionally
        cast the type.

    Expects the following format:
    "fake_data": {
        "value": "true",
        "section": "bci_config",
        "name": "Fake Data Sessions",
        "helpTip": "If true, fake data server used",
        "recommended": "",
        "editable": "true",
        "type": "bool"
        }

    PARAMETERS
    ----------
    :param: path: string path to the parameters file.
    :param: value_case: True/False cast values to specified type.

    Returns
    -------
        a Parameters object that behaves like a dict.
    """
    return Parameters(source=path, cast_values=value_cast)


def load_experimental_data() -> str:
    filename = ask_directory()  # show dialog box and return the path
    log.info("Loaded Experimental Data From: %s" % filename)
    return filename


def load_signal_models(directory: Optional[str] = None) -> List[SignalModel]:
    """Load all signal models in a given directory.

    Models are assumed to have been written using bcipy.helpers.save.save_model
    function and should be serialized as pickled files. Note that reading
    pickled files is a potential security concern so only load from trusted
    directories.

    Args:
        dirname (str, optional): Location of pretrained models. If not
            provided the user will be prompted for a location.
    """
    if not directory or Path(directory).is_file():
        directory = ask_directory()

    # update preferences
    path = Path(directory)
    preferences.signal_model_directory = str(path)

    models = []
    for file_path in path.glob(f"*{SIGNAL_MODEL_FILE_SUFFIX}"):
        with open(file_path, "rb") as signal_file:
            model = pickle.load(signal_file)
            log.info(f"Loading model {model}")
            models.append(model)
    return models


def choose_signal_models(device_types: List[str]) -> List[SignalModel]:
    """Prompt the user to load a signal model for each provided device.

    Parameters
    ----------
        device_types - list of device content types (ex. 'EEG')
    """
    return [
        model for model in map(choose_signal_model, set(device_types)) if model
    ]


def load_signal_model(file_path: str) -> SignalModel:
    """Load signal model from persisted file.

    Models are assumed to have been written using bcipy.io.save.save_model
    function and should be serialized as pickled files. Note that reading
    pickled files is a potential security concern so only load from trusted
    directories."""

    with open(file_path, "rb") as signal_file:
        model = pickle.load(signal_file)
        log.info(f"Loading model {model}")
        return model


def choose_signal_model(device_type: str) -> Optional[SignalModel]:
    """Present a file dialog prompting the user to select a signal model for
    the given device.

    Parameters
    ----------
        device_type - ex. 'EEG' or 'Eyetracker'; this should correspond with
            the content_type of the DeviceSpec of the model.
    """

    file_path = ask_filename(file_types=f"*{SIGNAL_MODEL_FILE_SUFFIX}",
                             directory=preferences.signal_model_directory,
                             prompt=f"Select the {device_type} signal model")

    if file_path:
        # update preferences
        path = Path(file_path)
        preferences.signal_model_directory = str(path)
        return load_signal_model(str(path))
    return None


def choose_csv_file(filename: Optional[str] = None) -> Optional[str]:
    """GUI prompt to select a csv file from the file system.

    Parameters
    ----------
    - filename : optional filename to use; if provided the GUI is not shown.

    Returns
    -------
    file name of selected file; throws an exception if the file is not a csv.
    """
    if not filename:
        filename = ask_filename('*.csv')

    # get the last part of the path to determine file type
    file_name = filename.split('/')[-1]

    if 'csv' not in file_name:
        raise Exception(
            'File type unrecognized. Please use a supported csv type')

    return filename


def load_raw_data(filename: Union[Path, str]) -> RawData:
    """Reads the data (.csv) file written by data acquisition.

    Parameters
    ----------
    - filename : path to the serialized data (csv file)

    Returns
    -------
    RawData object with data held in memory
    """
    return RawData.load(filename)


def load_users(data_save_loc: str) -> List[str]:
    """Load Users.

    Loads user directory names below experiments from the data path defined and returns them as a list.
    If the save data directory is not found, this method returns an empty list assuming no experiments
    have been run yet.
    """
    try:
        bcipy_data = BciPyCollection(data_directory=data_save_loc)
        bcipy_data.load_users()
    except FileNotFoundError:
        return []
    return bcipy_data.users


def fast_scandir(directory_name: str, return_path: bool = True) -> List[str]:
    """Fast Scan Directory.

    directory_name: name of the directory to be scanned
    return_path: whether or not to return the scanned directories as a relative path or name.
        False will return the directory name only.
    """
    if return_path:
        return [f.path for f in os.scandir(directory_name) if f.is_dir()]

    return [f.name for f in os.scandir(directory_name) if f.is_dir()]


class BciPySessionTaskData:
    """Session Task Data.

    This class is used to represent a single task data session. It is used to store the
    path to the task data, as well as the parameters and other information about the task.

    /<local_path>/<user_id>/<date>/<experiment_id>/<date_time>/
        protocol.json
        <task_date_time>/
            parameters.json
            **task_data**

    """

    def __init__(
            self,
            path: str,
            user_id: str,
            experiment_id: str,
            session_id: Optional[str] = None,
            task_name: Optional[str] = None,
            run: int = 1) -> None:

        self.user_id = user_id
        self.experiment_id = experiment_id.replace('_', '')
        self.session_id = session_id
        self.run = str(run)
        self.path = path
        self.task_name = task_name
        self.info = {
            'user_id': user_id,
            'experiment_id': self.experiment_id,
            'task_name': self.task_name,
            'run': run,
            'path': path
        }

    def get_parameters(self) -> Parameters:
        return load_json_parameters(
            f'{self.path}/{DEFAULT_PARAMETERS_FILENAME}',
            value_cast=True)

    def __str__(self):
        return f'BciPySessionTaskData: {self.info=}'

    def __repr__(self):
        return f'BciPySessionTaskData: {self.info=}'


class BciPyCollection:
    """BciPy Data.

    This class is used to represent a full BciPy data collection. It is used to collect data from the
    data directory and filter based on the provided filters.
    """

    def __init__(
            self,
            data_directory: str,
            experiment_id_filter: Optional[str] = None,
            user_id_filter: Optional[str] = None,
            date_filter: Optional[str] = None,
            date_time_filter: Optional[str] = None,
            excluded_tasks: Optional[List[str]] = None,
            anonymize: bool = False) -> None:
        if not os.path.isdir(data_directory):
            raise FileNotFoundError(
                f'Data directory not found at [{data_directory}]')
        self.data_directory = data_directory

        self.experiment_id_filter = experiment_id_filter
        self.user_id_filter = user_id_filter
        self.date_filter = date_filter
        self.date_time_filter = date_time_filter
        self.excluded_tasks = excluded_tasks
        self.anonymize = anonymize

        # Initialize the data lists
        self.session_task_data: List[BciPySessionTaskData] = []
        self.user_paths: List[str] = []
        self.date_paths: List[str] = []
        self.experiment_paths: List[str] = []
        self.date_time_paths: List[str] = []
        self.task_paths: List[str] = []

    def __repr__(self):
        return f'BciPyCollection: {self.data_directory=}'

    def __str__(self):
        return f'BciPyCollection: {self.data_directory=}'

    @property
    def users(self) -> List[str]:
        return [user.split('/')[-1] for user in self.user_paths]

    @property
    def experiments(self) -> List[str]:
        experiments = [experiment.split('/')[-1] for experiment in self.experiment_paths]
        # remove duplicates from the list
        return list(set(experiments))

    @property
    def dates(self) -> List[str]:
        dates = [date.split('/')[-1] for date in self.date_paths]
        # remove duplicates from the list
        return list(set(dates))

    @property
    def date_times(self) -> List[str]:
        date_times = [date_time.split('/')[-1] for date_time in self.date_time_paths]
        # remove duplicates from the list
        return list(set(date_times))

    @property
    def tasks(self) -> List[str]:
        tasks = [task.task_name for task in self.session_task_data]
        # remove duplicates from the list
        return list(set(tasks))

    def collect(self) -> List[BciPySessionTaskData]:
        """Collect.

        Collects the BciPy data from the data directory and returns a list of BciPySessionTaskData objects.
        """
        if not self.session_task_data:
            self.load_tasks()

        # if anonymize is set, return the anonymized data. Make a map of user_id to anonymized id
        if self.anonymize:
            user_map = {}
            user_id_increment = 1
            for task in self.session_task_data:
                if task.user_id not in user_map:
                    user_map[task.user_id] = f'ID{user_id_increment}'
                    user_id_increment += 1
                task.user_id = user_map[task.user_id]
            log.info(f"Anonymized user ids: {user_map}")

        return self.session_task_data

    def load_users(self) -> None:
        """Load Users.

        Walks the data directory and sets the user paths. It will filter by the user id if provided.
        """
        user_paths = fast_scandir(self.data_directory, return_path=True)
        if self.user_id_filter:
            self.user_paths = [user for user in user_paths if self.user_id_filter in user]
        else:
            self.user_paths = user_paths

    def load_dates(self) -> None:
        """Load Dates.

        Walks the data directory and sets the date paths. It will filter by the date if provided.
        """
        if not self.user_paths:
            self.load_users()

        for user in self.user_paths:
            data_paths = fast_scandir(user, return_path=True)
            if self.date_filter:
                self.date_paths.extend([data for data in data_paths if self.date_filter in data])
            else:
                self.date_paths.extend(data_paths)

    def load_experiments(self) -> None:
        """Load Experiments.

        Walks the data directory and sets the experiment paths. It will filter by the experiment id if provided.
        """
        if not self.date_paths:
            self.load_dates()

        for date in self.date_paths:
            experiment_paths = fast_scandir(date, return_path=True)
            if self.experiment_id_filter:
                self.experiment_paths.extend([data for data in experiment_paths if self.experiment_id_filter in data])
            else:
                self.experiment_paths.extend(experiment_paths)

    def load_date_times(self) -> None:
        """Load Date Times.

        Walks the data directory and sets the date time paths. It will filter by the date time if provided.
        """
        if not self.experiment_paths:
            self.load_experiments()

        for experiment in self.experiment_paths:
            data_paths = fast_scandir(experiment, return_path=True)
            if self.date_time_filter:
                self.date_time_paths.extend([data for data in data_paths if self.date_time_filter in data])
            else:
                self.date_time_paths.extend(data_paths)

    def sort_tasks(self, tasks: List[str]) -> List[str]:
        """Sort Tasks.

        Sorts the tasks in the order they were run using the timestamp at the end of the task path.
        """
        return sorted(tasks, key=lambda x: x.split('_')[-1])

    def load_tasks(self) -> None:
        """Load Tasks.

        Walks the data directory and sets the session_task_data representing the experiment data.
        It will exclude tasks that are in the excluded_tasks list.
        """
        if not self.date_time_paths:
            self.load_date_times()

        for date_time in self.date_time_paths:
            tasks = fast_scandir(date_time, return_path=True)
            run = 1
            tasks = self.sort_tasks(tasks)
            for task in tasks:
                task_path = Path(task)
                task = task_path.parts[-1]

                # skip excluded tasks
                for excluded_task in self.excluded_tasks:
                    if excluded_task in task:
                        log.info(f'Skipping excluded task [{task}]')
                        skip = True
                        break
                    else:
                        skip = False

                if skip:
                    continue

                user_id = task_path.parts[-5]
                date = task_path.parts[-4]
                experiment_id = task_path.parts[-3]
                date_time = task_path.parts[-2]
                self.session_task_data.append(
                    BciPySessionTaskData(
                        path=task_path,
                        user_id=user_id,
                        date=date,
                        experiment_id=experiment_id,
                        date_time=date_time,
                        run=run,
                        task=task
                    )
                )
                run += 1

        self.task_paths = [task.path for task in self.session_task_data]


def load_bcipy_data(
        data_directory: str,
        experiment_id: Optional[str] = None,
        user_id: Optional[str] = None,
        date: Optional[str] = None,
        date_time: Optional[str] = None,
        excluded_tasks: Optional[List[str]] = None,
        anonymize: bool = False) -> List[BciPySessionTaskData]:
    """Load BciPy Data.

    Walks a data directory and returns a list of data paths for the given experiment id, user id, and date.

    The BciPy data directory is structured as follows:
    data/
        user_ids/
            dates/
                experiment_ids/
                    datetimes/
                        protocol.json
                        logs/
                        tasks/
                            raw_data.csv
                            triggers.txt

    data_directory: the bcipy data directory to walk
    experiment_id: the experiment id to filter by
    user_id: the user id to filter by
    date: the date to filter by
    date_time: the date time to filter by
    excluded_tasks: a list of tasks to exclude from the returned list of experiment data
    anonymize: whether or not to anonymize the user ids

    Returns:
    --------
    a list of BciPySessionTaskData objects representing the experiment data
    """
    if not excluded_tasks:
        excluded_tasks = []

    # add logs to the excluded tasks
    excluded_tasks.append('logs')

    bcipy_data = BciPyCollection(
        data_directory=data_directory,
        experiment_id_filter=experiment_id,
        user_id_filter=user_id,
        date_filter=date,
        date_time_filter=date_time,
        excluded_tasks=excluded_tasks,
        anonymize=anonymize
    )
    return bcipy_data.collect()
