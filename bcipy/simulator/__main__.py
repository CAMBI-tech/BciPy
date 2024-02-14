""" Entry point to run Simulator """

import argparse
import datetime
import logging
import os
import random
import sys
from pathlib import Path

from bcipy.simulator.helpers.decision import MaxIterationsSim
from bcipy.simulator.sim_factory import SimulationFactoryV2
from bcipy.simulator.simulator_base import Simulator


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
    # creating wrapper dir to save to within /generated. Adds a unique 4 digit id to end
    unique_id = random.sample(range(1000, 10000), 1)[0]
    save_dir = f"{output_path}/SIM_{save_dir_name}_{unique_id}"
    os.makedirs(save_dir)
    os.makedirs(f"{save_dir}/logs")

    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_folder",
        type=Path,
        required=True,
        help="Raw data folders to be processed. Singular wrapper dir with data folders")
    parser.add_argument(
        "-sm",
        "--smodel_files",
        action="append",
        type=Path,
        required=True,
        help="Signal models to be used")

    parser.add_argument(
        "-p",
        "--parameters",
        type=Path,
        required=True,
        help="Parameter File to be used")

    parser.add_argument("-o", "--out_dir", type=Path, default=Path(__file__).resolve().parent)

    args = vars(parser.parse_args())

    output_path = "bcipy/simulator/generated"  # TODO read from parameters
    now_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
    args['save_dir'] = init_save_dir(output_path, now_time)

    # setting up logging
    log_path = f"{args['save_dir']}/logs"
    configure_logger(log_path, now_time)

    simulator: Simulator = SimulationFactoryV2.create(**args)

    simulator.run()
