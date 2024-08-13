""" Handles artifacts related logic ie. logs, save dir creation, result.json, ..."""
import logging
import os
import random
import sys
from typing import Optional


def configure_logger(log_path: str,
                     file_name: str,
                     logger_name: Optional[str] = None,
                     use_stdout: bool = True):
    """Configures logger for standard out and file output.
    
    Parameters
    ----------
        log_path - directory in which log file will reside
        file_name - name of the log file
        logger_name - None configures the root logger; otherwise configures a named logger.
        use_stdout - if True, INFO messages will be output to stdout.
    """

    log = logging.getLogger(logger_name)  # configuring root logger
    log.setLevel(logging.DEBUG)
    # Create handlers for logging to the standard output and a file
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    file_handler = logging.FileHandler(f"{log_path}/{file_name}")

    # Set the log levels on the handlers
    stdout_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    fmt = '[%(asctime)s][%(name)s][%(levelname)s]: %(message)s'
    fmt_file = logging.Formatter(fmt)
    fmt_stdout = logging.Formatter("%(message)s")

    # Set the log format on each handler
    stdout_handler.setFormatter(fmt_stdout)
    file_handler.setFormatter(fmt_file)

    # Add each handler to the Logger object
    if use_stdout and stdout_handler not in log.handlers:
        log.addHandler(stdout_handler)

    log.addHandler(file_handler)


def init_save_dir(output_path, save_dir_name):
    """ creating wrapper dir to save sim results to.
     - Saves to {output_path}
     - Adds a unique 4-digit id to end
     """

    unique_id = random.sample(range(1000, 10000), 1)[0]
    save_dir = f"{output_path}/SIM_{save_dir_name}_{unique_id}"
    os.makedirs(save_dir)
    os.makedirs(f"{save_dir}/logs")

    return save_dir
