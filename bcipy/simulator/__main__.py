""" Entry point to run Simulator """

import argparse
import datetime
import logging
import sys
from pathlib import Path

from bcipy.simulator.helpers.decision import MaxIterationsSim
from bcipy.simulator.sim_factory import SimulationFactoryV2
from bcipy.simulator.simulator_base import Simulator


def configure_logger():
    """ configures logger for standard out nad file output """

    log = logging.getLogger(None)  # configuring root logger
    log.setLevel(logging.DEBUG)
    # Create handlers for logging to the standard output and a file
    stdoutHandler = logging.StreamHandler(stream=sys.stdout)
    file_name = datetime.datetime.now().strftime("%m-%d-%H:%M")
    output_path = "bcipy/simulator/generated"
    fileHandler = logging.FileHandler(f"{output_path}/{file_name}.log")

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

    # setting up logging
    configure_logger()

    simulator: Simulator = SimulationFactoryV2.create(**args)

    simulator.run()
