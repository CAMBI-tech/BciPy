"""Setup and run the task"""

import argparse
import logging
import sys
from pathlib import Path

# pylint: disable=unused-import
# flake8: noqa
from bcipy.simulator.data.sampler import Sampler, TargetNontargetSampler
from bcipy.simulator.task.copy_phrase import SimulatorCopyPhraseTask
from bcipy.simulator.task.task_factory import TaskFactory
from bcipy.simulator.ui import cli, gui
from bcipy.simulator.util.artifact import (DEFAULT_SAVE_LOCATION,
                                           TOP_LEVEL_LOGGER_NAME,
                                           configure_run_directory,
                                           init_simulation_dir)

logger = logging.getLogger(TOP_LEVEL_LOGGER_NAME)


def classify(classname):
    """Convert the given class name to a class."""
    return getattr(sys.modules[__name__], classname)


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
        help="Use interactive command line for selecting simulator inputs")
    parser.add_argument(
        "--gui",
        required=False,
        default=False,
        action='store_true',
        help="Use interactive GUI for selecting simulator inputs")
    parser.add_argument("-d",
                        "--data_folder",
                        type=Path,
                        required=False,
                        action='append',
                        help=data_help)
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
    parser.add_argument("-s",
                        "--sampler",
                        type=str,
                        required=False,
                        default='TargetNontargetSampler',
                        help="Sampling strategy")
    parser.add_argument("-o",
                        "--output",
                        type=Path,
                        required=False,
                        default=DEFAULT_SAVE_LOCATION,
                        help="Number of times to run the simulation")
    args = parser.parse_args()
    sim_args = vars(args)

    runs = sim_args['n']
    outdir = sim_args['output']

    if args.gui:
        runs, outdir, task_factory = gui.configure()
    elif args.interactive:
        task_factory = cli.main(sim_args)
    else:
        task_factory = TaskFactory(params_path=sim_args['parameters'],
                                   source_dirs=sim_args['data_folder'],
                                   signal_model_paths=sim_args['model_path'],
                                   sampling_strategy=classify(sim_args['sampler']),
                                   task=SimulatorCopyPhraseTask)

    if task_factory:
        sim_dir = init_simulation_dir(save_location=outdir)
        logger.info(sim_dir)

        runner = TaskRunner(save_dir=sim_dir,
                            task_factory=task_factory,
                            runs=runs)
        runner.run()


if __name__ == '__main__':
    main()
