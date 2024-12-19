""" Handles artifacts related logic ie. logs, save dir creation, result.json, ..."""
import datetime
import logging
import os
import sys
from typing import Optional

from bcipy.config import ROOT

# For a simulation, two loggers are configured: a top level logger summarizing
# setup and progress, and a logger for each simulation run. The root logger is
# re-configured for each simulation run.
TOP_LEVEL_LOGGER_NAME = 'sim_logger'
DEFAULT_LOGFILE_NAME = 'sim.log'
DEFAULT_SAVE_LOCATION = f"{ROOT}/data/simulator"
RUN_PREFIX = "run_"


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
    log.setLevel(logging.INFO)
    # Create handlers for logging to the standard output and a file
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    file_handler = logging.FileHandler(f"{log_path}/{file_name}")

    # Set the log levels on the handlers
    stdout_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

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


def init_simulation_dir(save_location: str = DEFAULT_SAVE_LOCATION,
                        logfile_name: str = DEFAULT_LOGFILE_NAME) -> str:
    """Setup the folder structure and logging for a simulation.

    Parameters
    ----------
        save_location - optional path in which new simulation directory will be created.
        logfile_name - optional name of the top level logfile within that directory.

    Returns the path of the simulation directory.
    """
    save_dir = f"{save_location}/{directory_name()}"
    os.makedirs(save_dir)
    configure_logger(save_dir,
                     logfile_name,
                     logger_name=TOP_LEVEL_LOGGER_NAME,
                     use_stdout=True)
    return save_dir


def configure_run_directory(sim_dir: str, run: int) -> str:
    """Create the necessary directories and configure the logger.
    Returns the run directory.
    """
    run_name = f"{RUN_PREFIX}{run}"
    path = f"{sim_dir}/{run_name}"
    os.mkdir(path)
    configure_logger(log_path=path,
                     file_name=f"{run_name}.log",
                     use_stdout=False)
    return path


def directory_name() -> str:
    """Name of the directory for a new simulation run."""
    return datetime.datetime.now().strftime("SIM_%m-%d-%Y_%H_%M_%S")
