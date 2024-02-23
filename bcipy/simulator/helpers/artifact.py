""" Handles artifacts related logic ie. logs, save dir creation, result.json, ..."""
import logging
import os
import random
import sys


def configure_logger(log_path, file_name):
    """ configures logger for standard out nad file output """

    log = logging.getLogger(None)  # configuring root logger
    log.setLevel(logging.DEBUG)
    # Create handlers for logging to the standard output and a file
    stdoutHandler = logging.StreamHandler(stream=sys.stdout)
    fileHandler = logging.FileHandler(f"{log_path}/{file_name}.log")

    # Set the log levels on the handlers
    stdoutHandler.setLevel(logging.INFO)
    fileHandler.setLevel(logging.DEBUG)

    # Create a log format using Log Record attributes
    fmt_file = logging.Formatter("%(levelname)s | %(filename)s >> %(message)s")

    fmt = logging.Formatter("%(message)s")

    # Set the log format on each handler
    stdoutHandler.setFormatter(fmt)
    fileHandler.setFormatter(fmt_file)

    # Add each handler to the Logger object
    log.addHandler(stdoutHandler)
    log.addHandler(fileHandler)


def init_save_dir(output_path, save_dir_name):
    """ creating wrapper dir to save sim results to.
     - Saves to /generated
     - Adds a unique 4-digit id to end
     """

    unique_id = random.sample(range(1000, 10000), 1)[0]
    save_dir = f"{output_path}/SIM_{save_dir_name}_{unique_id}"
    os.makedirs(save_dir)
    os.makedirs(f"{save_dir}/logs")

    return save_dir
