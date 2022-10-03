import os

from bcipy.config import (
    DEFAULT_EXPERIMENT_PATH,
    DEFAULT_FIELD_PATH,
    EXPERIMENT_FILENAME,
    FIELD_FILENAME)
from bcipy.helpers.load import load_experiments, load_fields
from bcipy.helpers.system_utils import is_battery_powered, is_connected
from bcipy.helpers.exceptions import (InvalidFieldException,
                                      InvalidExperimentException,
                                      UnregisteredExperimentException,
                                      UnregisteredFieldException)
from bcipy.gui.alert import confirm


def validate_bcipy_session(parameters: dict) -> bool:
    """Check pre-conditions for a BciPy session. If any possible problems are
    detected, alert the user and prompt to continue.

    Parameters
    ----------
    parameters - configuration used to check for issues

    Returns
    -------
    True if it's okay to continue, otherwise False
    """
    possible_alerts = [(parameters['fake_data'], '* Fake data is on.'),
                       (is_connected(), '* Internet is on.'),
                       (is_battery_powered(), '* Operating on battery power')]
    alert_messages = [
        message for (condition, message) in possible_alerts if condition
    ]
    if alert_messages:
        lines = [
            "The following conditions may affect system behavior:\n",
            *alert_messages, "\nDo you want to continue?"
        ]
        return confirm("\n".join(lines))
    return True


def validate_experiment(
        experiment_name: str,
        experiment_path: str = f'{DEFAULT_EXPERIMENT_PATH}/{EXPERIMENT_FILENAME}',
        field_path: str = f'{DEFAULT_FIELD_PATH}/{FIELD_FILENAME}'
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

    # attempt to load the experiment by name
    _validate_experiment_format(experiment, experiment_name)
    _validate_experiment_fields(experiment['fields'], fields)

    return True


def _validate_experiment_fields(experiment_fields, fields):
    # loop over the experiment fields and attempt to load them by name
    for field in experiment_fields:
        # field is a dictionary with one key another dictionary as its' value
        # {'field_name': {require: bool, anonymize: bool}}
        field_name = list(field.keys())[0]
        try:
            fields[field_name]
        except KeyError:
            raise UnregisteredFieldException(f'Field [{field}] is not registered in [{fields}]')

        try:
            field[field_name]['required']
            field[field_name]['anonymize']
        except KeyError:
            raise InvalidFieldException(
                f'Experiment Field [{field}] incorrectly formatted. It should contain: required and anonymize')


def validate_field_data_written(path: str, file_name: str) -> bool:
    """Validate Field Data Written

    Validate that a field data file was written after executing the experiment field collection.
    """
    experiment_data_path = f'{path}/{file_name}'
    if os.path.isfile(experiment_data_path):
        return True
    raise InvalidFieldException(f'Experimental field data expected at path=[{experiment_data_path}] but not found.')


def validate_experiments(experiments, fields) -> bool:
    """Validate Experiments.

    Validate all experiments are in the correct format and the fields are properly registered.
    """
    for experiment_name in experiments:
        experiment = experiments[experiment_name]

        _validate_experiment_format(experiment, experiment_name)
        _validate_experiment_fields(experiment['fields'], fields)

    return True


def _validate_experiment_format(experiment, name):
    try:
        exp_summary = experiment['summary']
        assert isinstance(exp_summary, str)
        experiment_fields = experiment['fields']
        assert isinstance(experiment_fields, list)
    except KeyError:
        raise InvalidExperimentException(
            f'Experiment [{name}] is formatted incorrectly. It should contain the keys: summary and fields.')
    except AssertionError:
        raise InvalidExperimentException(
            f'Experiment [{name}] is formatted incorrectly. Unexpected type on summary(string) or fields(List[dict]).')
