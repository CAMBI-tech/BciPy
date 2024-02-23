""" Entry point to run Simulator """

import argparse
import datetime
from pathlib import Path

from bcipy.simulator.helpers import artifact
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

    output_path = "bcipy/simulator/generated"  # TODO read from parameters
    now_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
    args['save_dir'] = artifact.init_save_dir(output_path, now_time)

    # setting up logging
    log_path = f"{args['save_dir']}/logs"
    artifact.configure_logger(log_path, now_time)

    simulator: Simulator = SimulationFactoryV2.create(**args)

    simulator.run()
