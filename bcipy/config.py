"""Config.

This module contains the configuration for the BciPy application.
This includes the default parameters, static paths and core experiment configuration.
"""
from pathlib import Path

DEFAULT_ENCODING = 'utf-8'
DEFAULT_EVIDENCE_PRECISION = 5  # number of decimal places to round evidence to by default
MARKER_STREAM_NAME = 'TRG_device_stream'
DEFAULT_TRIGGER_CHANNEL_NAME = 'TRG'
DIODE_TRIGGER = '\u25A0'

# experiment configuration
DEFAULT_EXPERIMENT_ID = 'default'
DEFAULT_FRAME_RATE = 60
CUSTOM_TASK_EXPERIMENT_ID = "CustomTaskExecution"
EXPERIMENT_FILENAME = 'experiments.json'
FIELD_FILENAME = 'fields.json'
EXPERIMENT_DATA_FILENAME = 'experiment_data.json'
MULTIPHRASE_FILENAME = 'phrases.json'
PROTOCOL_FILENAME = 'protocol.json'
BCIPY_ROOT = Path(__file__).resolve().parent
ROOT = BCIPY_ROOT.parent
DEFAULT_EXPERIMENT_PATH = f'{BCIPY_ROOT}/parameters/experiment'
DEFAULT_FIELD_PATH = f'{BCIPY_ROOT}/parameters/field'
DEFAULT_USER_ID = 'test_user'
TASK_SEPERATOR = '->'

DEFAULT_PARAMETERS_FILENAME = 'parameters.json'
DEFAULT_DEVICES_PATH = f"{BCIPY_ROOT}/parameters"
DEFAULT_PARAMETERS_PATH = f'{BCIPY_ROOT}/parameters/{DEFAULT_PARAMETERS_FILENAME}'
DEFAULT_DEVICE_SPEC_FILENAME = 'devices.json'
DEVICE_SPEC_PATH = f'{BCIPY_ROOT}/parameters/{DEFAULT_DEVICE_SPEC_FILENAME}'
DEFAULT_LM_PARAMETERS_FILENAME = 'lm_params.json'
DEFAULT_LM_PARAMETERS_PATH = f'{BCIPY_ROOT}/parameters/{DEFAULT_LM_PARAMETERS_FILENAME}'

STATIC_PATH = f'{BCIPY_ROOT}/static'
STATIC_IMAGES_PATH = f'{STATIC_PATH}/images'
STATIC_AUDIO_PATH = f'{STATIC_PATH}/sounds'
BCIPY_LOGO_PATH = f'{STATIC_IMAGES_PATH}/gui/cambi.png'
BCIPY_FULL_LOGO_PATH = f'{STATIC_IMAGES_PATH}/gui/CAMBI_full_logo.png'
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
SESSION_LOG_FILENAME = 'session_log.txt'
PROTOCOL_LOG_FILENAME = 'protocol_log.txt'
STIMULI_POSITIONS_FILENAME = 'stimuli_positions.json'

# misc configuration
WAIT_SCREEN_MESSAGE = 'Press Space to start or Esc to exit'
MAX_PAUSE_SECONDS = 365
SESSION_COMPLETE_MESSAGE = 'Complete! Saving data...'
REMOTE_SERVER = "https://github.com/CAMBI-tech/BciPy/"
