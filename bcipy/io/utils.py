import re
from datetime import time


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
    filename_ex = "005_Matrix_Calibration_1716331369"

    pattern = r'^\d+_([A-Za-z]+(?:_[A-Za-z]+)*)_\d+$'
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        session_type = match.group(1)
        # Normalize the task type to uppercase
        session_type = session_type.upper()
        return session_type
    else:
        return default

def extract_timestamp(filename: str) -> time:
    """Extracts the timestamp (last number) from the filename."""
    match = re.search(r'(\d{1,2})hr(\d{1,2})min(\d{1,2})sec$', filename)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        timestamp = time(hour=hours, minute=minutes, second=seconds)
        return timestamp
    else:
        raise ValueError(f"No timestamp found in {filename}")

def sort_by_timestamp(file_list):
    """Sorts filenames by extracted timestamp."""
    return sorted(file_list, key=extract_timestamp)

if __name__ == "__main__":
    filenames = [
        "007_RSVP_CopyPhrase_IPO_14hr43min47sec",
        "007_RSVP_CopyPhrase_IPO_14hr25min31sec",
        "007_RSVP_CopyPhrase_IPO_14hr37min38sec",
        "007_RSVP_CopyPhrase_IPS_14hr50min36sec",
        "007_RSVP_CopyPhrase_IPO_14hr32min45sec",
    ]

    filenames_2 = [
        "0010_Matrix_Calibration_1715118261",
        "0010_RSVP_Calibration_1715122073",
        "0010_Matrix_CopyPhrase_1715119811",
        "0010_RSVP_CopyPhrase_1715122889",
        "0010_Matrix_CopyPhrase_1715120795",
        "0010_RSVP_CopyPhrase_1715123836",
    ]

    filenames_2 = sort_by_timestamp(filenames_2)
    for filename in filenames_2:
        task_type = extract_session_type(filename)
        print(f"Filename: {filename}, Session Type: {task_type}")