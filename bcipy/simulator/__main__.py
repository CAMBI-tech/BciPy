import argparse
from pathlib import Path

from bcipy.simulator.sim_factory import SimulationFactory, SimulationFactoryV2
from bcipy.simulator.simulator_base import Simulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_folders",
        action="append",
        type=Path,
        required=True,
        help="Raw data folders to be processed. This argument can be repeated to accumulate sessions data.")
    parser.add_argument(
        "-sm",
        "--smodel_files",
        action="append",
        type=Path,
        required=True,
        help="Signal models to be used")
    parser.add_argument("-o", "--out_dir", type=Path, default=Path(__file__).resolve().parent)

    args = vars(parser.parse_args())

    simulator: Simulator = SimulationFactoryV2.create(**args)
    simulator.run()
