"""Setup and run the task"""

import argparse
import logging
from glob import glob
from pathlib import Path

from bcipy.simulator.data.sampler import TargetNontargetSampler
from bcipy.simulator.task.copy_phrase import SimulatorCopyPhraseTask
from bcipy.simulator.task.task_factory import TaskFactory
from bcipy.simulator.ui import cli
from bcipy.simulator.util.artifact import (DEFAULT_SAVE_LOCATION,
                                           TOP_LEVEL_LOGGER_NAME,
                                           configure_run_directory,
                                           init_simulation_dir)

logger = logging.getLogger(TOP_LEVEL_LOGGER_NAME)


class TaskRunner():
    """Responsible for executing a task a given number of times."""

    def __init__(self,
                 save_dir: str,
                 task_factory: TaskFactory,
                 runs: int = 1):

        self.task_factory = task_factory
        self.sim_dir = save_dir
        self.runs = runs

    def run(self) -> None:
        """Run one or more simulations"""
        self.task_factory.parameters.save(self.sim_dir, 'parameters.json')
        for i in range(self.runs):
            logger.info(f"Executing task {i+1}")
            self.do_run(i + 1)
            logger.info("Task complete")

    def do_run(self, run: int):
        """Execute a simulation run."""
        run_dir = configure_run_directory(self.sim_dir, run)
        logger.info(run_dir)
        task = self.task_factory.make_task(run_dir)
        task.execute()


def main():
    """Run the task"""
    glob_help = ('glob pattern to select a subset of data folders'
                 ' Ex. "*RSVP_Copy_Phrase*"'
                 ' Used with a single data_folder')
    data_help = (
        'Raw data folders to be processed.'
        ' Multiple values can be provided, or a single parent folder.')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--interactive",
        required=False,
        default=False,
        action='store_true',
        help="Use interactive mode for selecting simulator inputs")
    parser.add_argument("-d",
                        "--data_folder",
                        type=Path,
                        required=False,
                        action='append',
                        help=data_help)
    parser.add_argument('-g', '--glob_pattern', help=glob_help, default="*")
    parser.add_argument(
        "-m",
        "--model_path",
        type=Path,
        action='append',
        required=False,
        help="Signal models to be used. Multiple models can be provided.")
    parser.add_argument("-p",
                        "--parameters",
                        type=Path,
                        required=False,
                        help="Parameter File to be used")
    parser.add_argument("-n",
                        type=int,
                        required=False,
                        default=1,
                        help="Number of times to run the simulation")
    parser.add_argument("-o",
                        "--output",
                        type=Path,
                        required=False,
                        default=DEFAULT_SAVE_LOCATION,
                        help="Number of times to run the simulation")
    args = parser.parse_args()
    sim_args = vars(args)

    sim_dir = init_simulation_dir(save_location=sim_args['output'])
    logger.info(sim_dir)

    if args.interactive:
        task_factory = cli.main(sim_args)
    else:
        source_dirs = sim_args['data_folder']
        if len(source_dirs) == 1:
            parent = source_dirs[0]
            source_dirs = [
                Path(d)
                for d in glob(str(Path(parent, sim_args['glob_pattern'])))
                if Path(d).is_dir()
            ]
        task_factory = TaskFactory(params_path=sim_args['parameters'],
                                   source_dirs=source_dirs,
                                   signal_model_paths=sim_args['model_path'],
                                   sampling_strategy=TargetNontargetSampler,
                                   task=SimulatorCopyPhraseTask)

    runner = TaskRunner(save_dir=sim_dir,
                        task_factory=task_factory,
                        runs=sim_args['n'])
    runner.run()


if __name__ == '__main__':
    main()
