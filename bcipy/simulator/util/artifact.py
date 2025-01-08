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

    # clear existing handlers
    handlers = log.handlers[:]  # make copy
    for handler in handlers:
        # call the method, which handles locking.
        log.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()

    log.setLevel(logging.INFO)
    fmt = '[%(asctime)s][%(name)s][%(levelname)s]: %(message)s'

    if use_stdout:
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(stdout_handler)

    file_handler = logging.FileHandler(f"{log_path}/{file_name}", delay=True)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt))
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
                     logger_name=None,
                     use_stdout=False)
    return path


def remove_file_logger(sim_dir: str, run: int) -> None:
    """Disable any configured file handler for the root logger."""
    path = f"{sim_dir}/{RUN_PREFIX}{run}"

    log = logging.getLogger()
    handlers = log.handlers[:]
    for handler in handlers:
        if isinstance(handler,
                      logging.FileHandler) and path in handler.baseFilename:
            log.removeHandler(handler)
            handler.close()


def directory_name() -> str:
    """Name of the directory for a new simulation run."""
    return datetime.datetime.now().strftime("SIM_%m-%d-%Y_%H_%M_%S")
