import doctest
import re


def extract_task_type(filename: str, default: str = "Unknown") -> str:
    filename_ex = "009_RSVP_CopyPhrase_IPO_11hr04min18sec"

    # Define valid task types
    task_types = ['IPO', 'IPS', 'NP', 'IPC']

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

if __name__ == "__main__":
    filenames = [
        "007_RSVP_CopyPhrase_IPO_14hr43min47sec",
        "007_RSVP_CopyPhrase_IPO_14hr25min31sec",
        "007_RSVP_CopyPhrase_IPO_14hr37min38sec",
        "007_RSVP_CopyPhrase_IPS_14hr50min36sec",
        "007_RSVP_CopyPhrase_IPO_14hr32min45sec",
    ]

    sorted = sorted(filenames)
    print(sorted)