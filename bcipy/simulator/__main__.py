""" Entry point to run Simulator """

import argparse
from glob import glob
from pathlib import Path

from bcipy.simulator.helpers.sim_runner import MultiSimRunner, SingleSimRunner
from bcipy.simulator.sim_factory import SimulationFactoryV2
from bcipy.simulator.simulator_base import Simulator

if __name__ == "__main__":
    glob_help = ('glob pattern to select a subset of data folders'
                 ' Ex. "*RSVP_Copy_Phrase*"')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_folder",
        type=Path,
        required=True,
        help="Raw data folders to be processed. Singular wrapper dir with data folders")
    parser.add_argument('-g', '--glob_pattern', help=glob_help, default="*")
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
    args = parser.parse_args()
    sim_args = vars(args)

    sim_args['source_dirs'] = [
        Path(d) for d in glob(str(Path(args.data_folder, args.glob_pattern)))
        if Path(d).is_dir()
    ]

    simulator: Simulator = SimulationFactoryV2.create(**sim_args)

    sim_run_count = simulator.get_parameters().get('sim_run_count', 1)

    if sim_run_count > 1:  # running multiple times
        runner = MultiSimRunner(simulator, sim_run_count)
        runner.run()
    else:
        runner = SingleSimRunner(simulator)
        runner.run()
