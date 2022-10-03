"""Config.

This module contains the configuration for the BciPy application.
This includes the default parameters, static paths and core experiment configuration.
"""
from pathlib import Path

DEFAULT_ENCODING = 'utf-8'

# experiment configuration
DEFAULT_EXPERIMENT_ID = 'default'
EXPERIMENT_FILENAME = 'experiments.json'
FIELD_FILENAME = 'fields.json'
EXPERIMENT_DATA_FILENAME = 'experiment_data.json'
BCIPY_ROOT = Path(__file__).resolve().parent
ROOT = BCIPY_ROOT.parent
DEFAULT_EXPERIMENT_PATH = f'{ROOT}/.bcipy/experiment'
DEFAULT_FIELD_PATH = f'{ROOT}/.bcipy/field'

DEFAULT_PARAMETER_FILENAME = 'parameters.json'
DEFAULT_PARAMETERS_PATH = f'{BCIPY_ROOT}/parameters/{DEFAULT_PARAMETER_FILENAME}'
DEFAULT_DEVICE_SPEC_FILENAME = 'devices.json'
DEVICE_SPEC_PATH = f'{BCIPY_ROOT}/parameters/{DEFAULT_DEVICE_SPEC_FILENAME}'

STATIC_PATH = f'{BCIPY_ROOT}/static'
STATIC_IMAGES_PATH = f'{STATIC_PATH}/images'
STATIC_AUDIO_PATH = f'{STATIC_PATH}/sounds'
BCIPY_LOGO_PATH = f'{STATIC_IMAGES_PATH}/gui/cambi.png'

# core data configuration
RAW_DATA_FILENAME = 'raw_data'
TRIGGER_FILENAME = 'triggers.txt'
SESSION_DATA_FILENAME = 'session.json'
SESSION_SUMMARY_FILENAME = 'session.xlsx'
LOG_FILENAME = 'bcipy_system_log.txt'

# misc configuration
WAIT_SCREEN_MESSAGE = 'Press Space to start or Esc to exit'
SESSION_COMPLETE_MESSAGE = 'Complete! Saving data...'
REMOTE_SERVER = "https://github.com/CAMBI-tech/BciPy/"
