""" Entry point to run Simulator """

import argparse
import datetime
from pathlib import Path

from bcipy.simulator.helpers.sim_runner import MultiSimRunner, SingleSimRunner
from bcipy.simulator.sim_factory import SimulationFactoryV2
from bcipy.simulator.simulator_base import Simulator

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

    simulator: Simulator = SimulationFactoryV2.create(**args)

    sim_run_count = simulator.get_parameters().get('sim_run_count', 1)

    if sim_run_count > 1:  # running multiple times
        runner = MultiSimRunner(simulator, sim_run_count)
        runner.run()
    else:
        runner = SingleSimRunner(simulator)
        runner.run()
