import argparse
from pathlib import Path

from bcipy.helpers.load import load_json_parameters
from bcipy.simulator.sim_factory import SimulationFactory

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_folders",
        action="append",
        type=Path,
        required=True,
        help="Session data folders to be processed. This argument can be repeated to accumulate sessions data.")
    parser.add_argument(
        "-sm",
        "--smodel_files",
        action="append",
        type=Path,
        required=True,
        help="Signal models to be used")
    parser.add_argument(
        "-lm",
        "--lmodel_file",
        action="append",
        type=Path,
        required=False,
        help="Language models to be used")
    parser.add_argument("-o", "--out_dir", type=Path, default=None)
    parser.add_argument(
        "-p",
        "--parameter_path",
        type=Path,
        default=None,
        help="Parameter file to be used for replay. If none, the session parameter file will be used.")

    args = vars(parser.parse_args())

    # assert len(set(args['data_folders'])) == len(args.data_folders), "Duplicated data folders"

    if args['out_dir'] is None:
        args['out_dir'] = Path(__file__).resolve().parent

    # Load parameters
    sim_parameters = load_json_parameters("bcipy/simulator/sim_parameters.json", value_cast=True)
    sim_task = sim_parameters.get("sim_task")
    args['sim_task'] = sim_task

    simulator = SimulationFactory.create(**args)
    simulator.run()
