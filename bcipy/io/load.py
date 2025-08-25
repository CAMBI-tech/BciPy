"""Module for loading BciPy data and configuration files.

This module provides functions for loading various types of data used in BciPy,
including parameters, experiments, signal models, and session data.
"""

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
                          EXPERIMENT_FILENAME, FIELD_FILENAME,
                          SESSION_LOG_FILENAME, SIGNAL_MODEL_FILE_SUFFIX)
from bcipy.core.parameters import Parameters
from bcipy.core.raw_data import RawData
from bcipy.exceptions import BciPyCoreException, InvalidExperimentException
from bcipy.gui.file_dialog import ask_directory, ask_filename
from bcipy.preferences import preferences
from bcipy.signal.model import SignalModel

log = logging.getLogger(SESSION_LOG_FILENAME)


def copy_parameters(path: str = DEFAULT_PARAMETERS_PATH,
                    destination: Optional[str] = None) -> str:
    """Creates a copy of the given configuration (parameters.json) to the given directory.

    Args:
        path: Optional path of parameters file to copy; uses default if not provided.
        destination: Optional destination directory; default is the same directory
            as the default parameters.

    Returns:
        str: Path to the new file.
    """
    default_dir = str(Path(DEFAULT_PARAMETERS_PATH).parent)

    destination = default_dir if destination is None else destination
    filename = strftime('parameters_%Y-%m-%d_%Hh%Mm%Ss.json', localtime())

    path = str(Path(destination, filename))
    copyfile(DEFAULT_PARAMETERS_PATH, path)
    return path


def load_experiments(path: str = f'{DEFAULT_EXPERIMENT_PATH}/{EXPERIMENT_FILENAME}') -> dict:
    """Load experiment configurations from a JSON file.

    Args:
        path: Path to the experiments file.

    Returns:
        dict: Dictionary of experiments with format:
            {
                name: {
                    fields: {
                        name: str,
                        required: bool,
                        anonymize: bool
                    },
                    summary: str
                }
            }
    """
    with open(path, 'r', encoding=DEFAULT_ENCODING) as json_file:
        return json.load(json_file)


def extract_mode(bcipy_data_directory: str) -> str:
    """Extract the task mode from a BciPy data save directory.

    This method extracts the task mode from a BciPy data save directory. This is important for
    trigger conversions and extracting targetness.

    Note:
        Not compatible with older versions of BciPy (pre 1.5.0) where
        the tasks and modes were considered together using integers (1, 2, 3).

    Args:
        bcipy_data_directory: Path to the data directory.

    Returns:
        str: The extracted mode ('calibration' or 'copy_phrase').

    Raises:
        BciPyCoreException: If no valid mode could be extracted.
    """
    directory = bcipy_data_directory.lower()
    if 'calibration' in directory:
        return 'calibration'
    elif 'copy' in directory:
        return 'copy_phrase'
    raise BciPyCoreException(
        f'No valid mode could be extracted from [{directory}]')


def load_fields(path: str = f'{DEFAULT_FIELD_PATH}/{FIELD_FILENAME}') -> dict:
    """Load field definitions from a JSON file.

    Args:
        path: Path to the fields file.

    Returns:
        dict: Dictionary of fields with format:
            {
                "field_name": {
                    "help_text": str,
                    "type": str
                }
            }
    """
    with open(path, 'r', encoding=DEFAULT_ENCODING) as json_file:
        return json.load(json_file)


def load_experiment_fields(experiment: dict) -> list:
    """Extract field names from an experiment configuration.

    Args:
        experiment: Dictionary containing experiment configuration with format:
            {
                'fields': [{field_dict}, {field_dict}],
                'summary': str
            }

    Returns:
        list: List of field names from the experiment configuration.

    Raises:
        InvalidExperimentException: If experiment format is incorrect.
        TypeError: If experiment is not a dictionary.
    """
    if isinstance(experiment, dict):
        try:
            return [name for field in experiment['fields'] for name in field.keys()]
        except KeyError:
            raise InvalidExperimentException(
                'Experiment is not formatted correctly. It should be passed as a dictionary with the fields and'
                f' summary keys. Fields is a list of dictionaries. Summary is a string. \n experiment=[{experiment}]')
    raise TypeError(
        'Unsupported experiment type. It should be passed as a dictionary with the fields and summary keys')


def load_json_parameters(path: str, value_cast: bool = False) -> Parameters:
    """Load and parse parameters from a JSON file.

    Args:
        path: Path to the parameters file.
        value_cast: Whether to cast values to their specified types.

    Returns:
        Parameters: A Parameters object containing the loaded configuration.

    Note:
        Expected JSON format:
        {
            "parameter_name": {
                "value": str,
                "section": str,
                "name": str,
                "helpTip": str,
                "recommended": str,
                "editable": str,
                "type": str
            }
        }
    """
    return Parameters(source=path, cast_values=value_cast)


def load_experimental_data(message='', strict=False) -> str:
    """Show a dialog to select an experimental data directory.

    Args:
        message: Optional prompt message for the dialog.
        strict: Whether to enforce strict directory selection.

    Returns:
        str: Path to the selected directory.
    """
    filename = ask_directory(prompt=message, strict=strict)
    log.info("Loaded Experimental Data From: %s" % filename)
    return filename


def load_signal_models(directory: Optional[str] = None) -> List[SignalModel]:
    """Load all signal models from a directory.

    Models are assumed to have been written using bcipy.helpers.save.save_model
    function and should be serialized as pickled files.

    Args:
        directory: Location of pretrained models. User will be prompted if not provided.

    Returns:
        list: List of loaded SignalModel instances.

    Warning:
        Reading pickled files is a potential security risk. Only load from trusted directories.
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

    Args:
        device_types: List of device content types (e.g., 'EEG').

    Returns:
        list: List of selected SignalModel instances.
    """
    return [
        model for model in map(choose_signal_model, set(device_types)) if model
    ]


def load_signal_model(file_path: str) -> SignalModel:
    """Load signal model from persisted file.

    Args:
        file_path: Path to the model file.

    Returns:
        SignalModel: The loaded signal model.

    Warning:
        Reading pickled files is a potential security risk. Only load from trusted sources.
    """
    with open(file_path, "rb") as signal_file:
        model = pickle.load(signal_file)
        log.info(f"Loading model {model}")
        return model


def choose_signal_model(device_type: str) -> Optional[SignalModel]:
    """Present a file dialog prompting the user to select a signal model.

    Args:
        device_type: Device type (e.g., 'EEG' or 'Eyetracker') that should correspond
            with the content_type of the DeviceSpec of the model.

    Returns:
        Optional[SignalModel]: The selected signal model, or None if no selection made.
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


def choose_model_paths(device_types: List[str]) -> List[Path]:
    """Select a model for each device and return a list of paths.

    Args:
        device_types: List of device types to load models for.

    Returns:
        list: List of paths to selected model files.
    """
    return [
        ask_filename(file_types=f"*{SIGNAL_MODEL_FILE_SUFFIX}",
                     directory=preferences.signal_model_directory,
                     prompt=f"Select the {device_type} signal model")
        for device_type in device_types
    ]


def choose_csv_file(filename: Optional[str] = None) -> Optional[str]:
    """GUI prompt to select a CSV file from the file system.

    Args:
        filename: Optional filename to use; if provided the GUI is not shown.

    Returns:
        Optional[str]: Path to selected file.

    Raises:
        Exception: If the selected file is not a CSV file.
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
    """Read data from a CSV file written by data acquisition.

    Args:
        filename: Path to the serialized data (CSV file).

    Returns:
        RawData: Object containing the loaded data in memory.
    """
    return RawData.load(filename)


def load_users(data_save_loc: str) -> List[str]:
    """Load user directory names from the data path.

    Args:
        data_save_loc: Path to the data directory.

    Returns:
        list: List of user IDs found in the directory. Returns empty list if
            directory not found (assuming no experiments have been run).
    """
    try:
        bcipy_data = BciPyCollection(data_directory=data_save_loc)
        bcipy_data.load_users()
    except FileNotFoundError:
        return []
    return bcipy_data.users


def fast_scandir(directory_name: str, return_path: bool = True) -> List[str]:
    """Quickly scan a directory for subdirectories.

    Args:
        directory_name: Name of the directory to scan.
        return_path: Whether to return full paths (True) or just names (False).

    Returns:
        list: List of subdirectory paths or names.
    """
    if return_path:
        return [f.path for f in os.scandir(directory_name) if f.is_dir()]

    return [f.name for f in os.scandir(directory_name) if f.is_dir()]


class BciPySessionTaskData:
    """Class representing data from a single BciPy task session.

    This class is used to store the path to the task data, as well as parameters
    and other information about the task.

    Directory structure:
        /<path>/
            protocol.json
            <task_date_time>/
                parameters.json
                **task_data**

    Args:
        path: Path to the session data.
        user_id: ID of the user who performed the task.
        experiment_id: ID of the experiment the task belongs to.
        date_time: Optional timestamp of task execution.
        date: Optional date of task execution.
        task_name: Optional name of the executed task.
        session_id: Session identifier number, defaults to 1.
        run: Run number within the session, defaults to 1.

    Attributes:
        user_id: ID of the user who performed the task.
        experiment_id: ID of the experiment (with underscores removed).
        session_id: Formatted session ID (zero-padded if < 10).
        date_time: Timestamp of task execution.
        date: Date of task execution.
        run: Formatted run number (zero-padded if < 10).
        path: Path to the session data.
        task_name: Name of the executed task.
        info: Dictionary containing all session information.
    """

    def __init__(
            self,
            path: str,
            user_id: str,
            experiment_id: str,
            date_time: Optional[str] = None,
            date: Optional[str] = None,
            task_name: Optional[str] = None,
            session_id: int = 1,
            run: int = 1) -> None:

        self.user_id = user_id
        self.experiment_id = experiment_id.replace('_', '')
        self.session_id = f'0{str(session_id)}' if session_id < 10 else str(
            session_id)
        self.date_time = date_time
        self.date = date
        self.run = f'0{str(run)}' if run < 10 else str(run)
        self.path = path
        self.task_name = task_name
        self.info = {
            'user_id': self.user_id,
            'experiment_id': self.experiment_id,
            'task_name': self.task_name,
            'session_id': self.session_id,
            'run': self.run,
            'date': self.date,
            'date_time': self.date_time,
            'path': self.path
        }

    def __str__(self):
        return f'BciPySessionTaskData: {self.info=}'

    def __repr__(self):
        return f'BciPySessionTaskData: {self.info=}'


class BciPyCollection:
    """Class for managing collections of BciPy session task data.

    This class is used to collect data from the data directory and filter based
    on the provided filters.

    Args:
        data_directory: Root directory containing BciPy data.
        experiment_id_filter: Optional filter for specific experiments.
        user_id_filter: Optional filter for specific users.
        date_filter: Optional filter for specific dates.
        date_time_filter: Optional filter for specific timestamps.
        excluded_tasks: Optional list of task names to exclude.
        anonymize: Whether to anonymize user data.

    Attributes:
        data_directory: Root directory containing BciPy data.
        experiment_id_filter: Filter for specific experiments.
        user_id_filter: Filter for specific users.
        date_filter: Filter for specific dates.
        date_time_filter: Filter for specific timestamps.
        excluded_tasks: List of task names to exclude.
        anonymize: Whether to anonymize user data.
        session_task_data: List of collected BciPySessionTaskData objects.
        user_paths: List of paths to user directories.
        date_paths: List of paths to date directories.
        experiment_paths: List of paths to experiment directories.
        date_time_paths: List of paths to datetime directories.
        task_paths: List of paths to task directories.
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
        """Get list of users in the collection.

        Returns:
            list: List of user IDs.
        """
        return [user.split('/')[-1] for user in self.user_paths]

    @property
    def experiments(self) -> List[str]:
        """Get list of unique experiments in the collection.

        Returns:
            list: List of experiment IDs.
        """
        experiments = [experiment.split('/')[-1]
                       for experiment in self.experiment_paths]
        # remove duplicates from the list
        return list(set(experiments))

    @property
    def dates(self) -> List[str]:
        """Get list of unique dates in the collection.

        Returns:
            list: List of dates.
        """
        dates = [date.split('/')[-1] for date in self.date_paths]
        # remove duplicates from the list
        return list(set(dates))

    @property
    def date_times(self) -> List[str]:
        """Get list of unique timestamps in the collection.

        Returns:
            list: List of timestamps.
        """
        date_times = [date_time.split('/')[-1]
                      for date_time in self.date_time_paths]
        # remove duplicates from the list
        return list(set(date_times))

    @property
    def tasks(self) -> List[str]:
        """Get list of unique tasks in the collection.

        Returns:
            list: List of task names.
        """
        tasks = [task.task_name for task in self.session_task_data]
        # remove duplicates from the list
        return list(set(tasks))

    def collect(self) -> List[BciPySessionTaskData]:
        """Collect BciPy data from the data directory.

        Returns:
            list: List of BciPySessionTaskData objects representing the experiment data.
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
        """Load user paths from the data directory.

        Walks the data directory and sets the user paths. Filters by user ID if provided.
        """
        user_paths = fast_scandir(self.data_directory, return_path=True)
        if self.user_id_filter:
            self.user_paths = [
                user for user in user_paths if self.user_id_filter in user]
        else:
            self.user_paths = user_paths

    def load_dates(self) -> None:
        """Load date paths from the data directory.

        Walks the data directory and sets the date paths. Filters by date if provided.
        """
        if not self.user_paths:
            self.load_users()

        for user in self.user_paths:
            data_paths = fast_scandir(user, return_path=True)
            if self.date_filter:
                self.date_paths.extend(
                    [data for data in data_paths if self.date_filter in data])
            else:
                self.date_paths.extend(data_paths)

    def load_experiments(self) -> None:
        """Load experiment paths from the data directory.

        Walks the data directory and sets the experiment paths. Filters by experiment ID if provided.
        """
        if not self.date_paths:
            self.load_dates()

        for date in self.date_paths:
            experiment_paths = fast_scandir(date, return_path=True)
            if self.experiment_id_filter:
                self.experiment_paths.extend(
                    [data for data in experiment_paths if self.experiment_id_filter in data])
            else:
                self.experiment_paths.extend(experiment_paths)

    def load_date_times(self) -> None:
        """Load datetime paths from the data directory.

        Walks the data directory and sets the datetime paths. Filters by datetime if provided.
        """
        if not self.experiment_paths:
            self.load_experiments()

        for experiment in self.experiment_paths:
            data_paths = fast_scandir(experiment, return_path=True)
            if self.date_time_filter:
                self.date_time_paths.extend(
                    [data for data in data_paths if self.date_time_filter in data])
            else:
                self.date_time_paths.extend(data_paths)

    def sort_tasks(self, tasks: List[str]) -> List[str]:
        """Sort tasks by their timestamp.

        Args:
            tasks: List of task paths to sort.

        Returns:
            list: Sorted list of task paths.
        """
        return sorted(tasks, key=lambda x: x.split('_')[-1])

    def load_tasks(self) -> None:
        """Load task data from the data directory.

        Walks the data directory and sets the session_task_data representing the experiment data.
        Excludes tasks that are in the excluded_tasks list.
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
                task_name = task.split(date)[0].strip('_')
                self.session_task_data.append(
                    BciPySessionTaskData(
                        path=task_path,
                        user_id=user_id,
                        date_time=date_time,
                        date=date,
                        experiment_id=experiment_id,
                        run=run,
                        task_name=task_name
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
    """Load BciPy data from a directory.

    Args:
        data_directory: The BciPy data directory to walk.
        experiment_id: Optional experiment ID to filter by.
        user_id: Optional user ID to filter by.
        date: Optional date to filter by.
        date_time: Optional datetime to filter by.
        excluded_tasks: Optional list of tasks to exclude.
        anonymize: Whether to anonymize user IDs.

    Returns:
        list: List of BciPySessionTaskData objects representing the experiment data.

    Note:
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
