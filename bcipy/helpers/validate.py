import os

from bcipy.helpers.load import load_experiments, load_fields, load_experiment_fields
from bcipy.helpers.system_utils import DEFAULT_EXPERIMENT_PATH, DEFAULT_FIELD_PATH, EXPERIMENT_FILENAME, FIELD_FILENAME
from bcipy.helpers.exceptions import (
    FieldException,
    UnregisteredExperimentException,
    UnregisteredFieldException
)


def validate_experiment(
        experiment_name: str,
        experiment_path: str = f'{DEFAULT_EXPERIMENT_PATH}{EXPERIMENT_FILENAME}',
        field_path: str = f'{DEFAULT_FIELD_PATH}{FIELD_FILENAME}'
) -> bool:
    """Validate Experiment.

    Validate the experiment is in the correct format and the fields are properly registered.
    """
    experiments = load_experiments(experiment_path)
    fields = load_fields(field_path)

    # attempt to load the experiment by name
    try:
        experiment = experiments[experiment_name]
    except KeyError:
        raise UnregisteredExperimentException(
            f'Experiment [{experiment_name}] is not registered at path [{experiment_path}]')

    # grab all field names as a list of strings. This call will raise exceptions if formatted incorrectly.
    experiment_fields = load_experiment_fields(experiment)

    # loop over the experiment fields and attempt to load them by name
    for field in experiment_fields:
        try:
            fields[field]
        except KeyError:
            raise UnregisteredFieldException(f'Field [{field}] is not registered at path [{field_path}]')

    return True


def validate_field_data_written(path: str, file_name: str) -> bool:
    """Validate Field Data Written

    Validate that a field data file was written after executing the experiment field collection.
    """
    experiment_data_path = f'{path}/{file_name}'
    if os.path.isfile(experiment_data_path):
        return True
    raise FieldException(f'Experimental field data expected at path=[{experiment_data_path}] but not found.')
