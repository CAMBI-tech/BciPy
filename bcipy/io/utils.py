import json
import re
from datetime import time, datetime
from pathlib import PosixPath


def extract_task_type(filename: str, default: str = "Unknown") -> str:
    filename_ex = "009_RSVP_CopyPhrase_IPO_11hr04min18sec"

    # Define valid task types
    task_types = ['IPO', 'IPS', 'NP', 'IPC', 'calibration']

    # Create a regex pattern from the task types
    pattern = r'_(' + '|'.join(task_types) + r')_'

    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        task_type = match.group(1)
        # Normalize the task type to uppercase
        task_type = task_type.upper()
        return task_type
    else:
        return default


def extract_session_type(filename: str, default: str = "Unknown") -> str:
    filename_ex = "/Users/srikarananthoju/cambi/data/jitter/0012/0012_RSVP_Calibration_10hr12min09sec"

    # pattern = r'^\d+_([A-Za-z]+(?:_[A-Za-z]+)*)_\d+$'
    # pattern = r'_(\w+_\w+)_\d+$'
    pattern = r'\d+_RSVP_([A-Za-z]+)_+'
    match = re.search(pattern, str(filename), re.IGNORECASE)
    if match:
        session_type = match.group(1)
        # Normalize the task type to uppercase
        session_type = session_type.replace('_', '')
        session_type = session_type.upper()
        return session_type
    else:
        return default


def extract_timestamp_jitter(filename: str) -> time:
    """Extracts the timestamp (last number) from the filename."""
    filename = str(filename)
    match = re.search(r'(\d{1,2})hr(\d{1,2})min(\d{1,2})sec$', filename)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        timestamp = time(hour=hours, minute=minutes, second=seconds)
        return timestamp
    else:
        raise ValueError(f"No timestamp found in {filename}")

def extract_timestamp_sspi(filename: str) -> datetime:
    ex = ["006_Matrix_Calibration_1728937425", "006_Matrix_CopyPhrase_1728938187"]
    pattern = r'_(\d+)$'
    match = re.search(pattern, str(filename))
    if match:
        match = int(match.group(1))
    else:
        raise ValueError(f"No timestamp found in {filename}")
    dt: datetime = datetime.fromtimestamp(match)
    return dt

def extract_jitter_type(parent_dir: PosixPath) -> str:
    """Extracts the jitter type from the parameters.json in filename"""

    parameters_file = parent_dir / "parameters.json"
    with open(parameters_file, 'r') as file:
        data = json.load(file)
        jitter_type = data.get('stim_jitter')

    if not jitter_type:
        raise ValueError(f"No jitter type found in {parameters_file}")

    return str(jitter_type["value"])

def sort_by_timestamp(file_list):
    """Sorts filenames by extracted timestamp."""
    return sorted(file_list, key=extract_timestamp_sspi)