"""Config.

This module contains the configuration for the BciPy application.
This includes the default parameters, static paths and core experiment configuration.
"""
from pathlib import Path

DEFAULT_ENCODING = 'utf-8'
DEFAULT_EVIDENCE_PRECISION = 5  # number of decimal places to round evidence to by default
MARKER_STREAM_NAME = 'TRG_device_stream'

# experiment configuration
DEFAULT_EXPERIMENT_ID = 'default'
EXPERIMENT_FILENAME = 'experiments.json'
FIELD_FILENAME = 'fields.json'
EXPERIMENT_DATA_FILENAME = 'experiment_data.json'
BCIPY_ROOT = Path(__file__).resolve().parent
ROOT = BCIPY_ROOT.parent
DEFAULT_EXPERIMENT_PATH = f'{BCIPY_ROOT}/parameters/experiment'
DEFAULT_FIELD_PATH = f'{BCIPY_ROOT}/parameters/field'

DEFAULT_PARAMETER_FILENAME = 'parameters.json'
DEFAULT_PARAMETERS_PATH = f'{BCIPY_ROOT}/parameters/{DEFAULT_PARAMETER_FILENAME}'
DEFAULT_DEVICE_SPEC_FILENAME = 'devices.json'
DEVICE_SPEC_PATH = f'{BCIPY_ROOT}/parameters/{DEFAULT_DEVICE_SPEC_FILENAME}'
DEFAULT_LM_PARAMETERS_FILENAME = 'lm_params.json'
DEFAULT_LM_PARAMETERS_PATH = f'{BCIPY_ROOT}/parameters/{DEFAULT_LM_PARAMETERS_FILENAME}'

STATIC_PATH = f'{BCIPY_ROOT}/static'
STATIC_IMAGES_PATH = f'{STATIC_PATH}/images'
STATIC_AUDIO_PATH = f'{STATIC_PATH}/sounds'
BCIPY_LOGO_PATH = f'{STATIC_IMAGES_PATH}/gui/cambi.png'
PREFERENCES_PATH = f'{ROOT}/bcipy_cache'
LM_PATH = f'{BCIPY_ROOT}/language/lms'
SIGNAL_MODEL_FILE_SUFFIX = '.pkl'

DEFAULT_FIXATION_PATH = f'{STATIC_IMAGES_PATH}/main/PLUS.png'
DEFAULT_TEXT_FIXATION = '+'

MATRIX_IMAGE_FILENAME = 'matrix.png'
DEFAULT_GAZE_IMAGE_PATH = f'{STATIC_IMAGES_PATH}/main/{MATRIX_IMAGE_FILENAME}'

# core data configuration
RAW_DATA_FILENAME = 'raw_data'
EYE_TRACKER_FILENAME_PREFIX = 'eyetracker_data'
TRIGGER_FILENAME = 'triggers.txt'
SESSION_DATA_FILENAME = 'session.json'
SESSION_SUMMARY_FILENAME = 'session.xlsx'
LOG_FILENAME = 'bcipy_system_log.txt'
STIMULI_POSITIONS_FILENAME = 'stimuli_positions.json'

# misc configuration
WAIT_SCREEN_MESSAGE = 'Press Space to start or Esc to exit'
MAX_PAUSE_SECONDS = 365
SESSION_COMPLETE_MESSAGE = 'Complete! Saving data...'
REMOTE_SERVER = "https://github.com/CAMBI-tech/BciPy/"
