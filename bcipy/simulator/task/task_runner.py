"""Setup and run the task"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from rich.progress import track

from bcipy.io.load import load_json_parameters
import bcipy.simulator.util.metrics as metrics
# pylint: disable=wildcard-import,unused-wildcard-import
# flake8: noqa
from bcipy.simulator.data.sampler import *
from bcipy.simulator.task.copy_phrase import SimulatorCopyPhraseTask
from bcipy.simulator.task.task_factory import TaskFactory
from bcipy.simulator.ui import cli, gui
from bcipy.simulator.util.artifact import (DEFAULT_SAVE_LOCATION,
                                           TOP_LEVEL_LOGGER_NAME,
                                           configure_run_directory,
                                           init_simulation_dir,
                                           remove_handlers, set_verbose)

logger = logging.getLogger(TOP_LEVEL_LOGGER_NAME)


def classify(classname):
    """Convert the given class name to a class."""
    return getattr(sys.modules[__name__], classname)


def parse_args(args: str) -> Dict[str, Any]:
    """Converts sampler command line args to a dictionary of parameters to be
    passed to the constructor.

    Parameters
    ----------
        args - str formatted as a valid JSON object.
    """
    return json.loads(args)


class TaskRunner():
    """Responsible for executing a task a given number of times."""

    def __init__(self,
                 save_dir: str,
                 task_factory: TaskFactory,
                 runs: int = 1,
                 verbose: bool = True):

        self.task_factory = task_factory
        self.sim_dir = save_dir
        self.runs = runs
        self.verbose = verbose

    def run(self) -> None:
        """Run one or more simulations"""
        self.task_factory.parameters.save(self.sim_dir, 'parameters.json')
        for i in track(range(self.runs), description="Processing..."):
            self.do_run(i + 1)

    def do_run(self, run: int):
        """Execute a simulation run."""
        logger.debug(f"Executing task {run}")
        run_dir, run_log = configure_run_directory(self.sim_dir, run)
        logger.debug(run_dir)
        task = self.task_factory.make_task(run_dir)
        task.execute()
        logger.debug("Task complete")
        remove_handlers(run_log)


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
    parser.add_argument("--sampler_args",
                        type=str,
                        required=False,
                        default="{}",
                        help="Sampler args structured as a JSON string.")
    parser.add_argument("-o",
                        "--output",
                        type=Path,
                        required=False,
                        default=DEFAULT_SAVE_LOCATION,
                        help="Sim output path")
    parser.add_argument("-v",
                        "--verbose",
                        action='store_true',
                        help="Verbose mode for more detailed logging.")
    args = parser.parse_args()
    sim_args = vars(args)

    runs = sim_args['n']
    outdir = sim_args['output']

    if args.gui:
        runs, outdir, task_factory = gui.configure()
    elif args.interactive:
        task_factory = cli.main(sim_args)
    else:
        parameters = load_json_parameters(sim_args['parameters'], value_cast=True)
        task_factory = TaskFactory(
            parameters=parameters,
            source_dirs=sim_args['data_folder'],
            signal_model_paths=sim_args['model_path'],
            sampling_strategy=classify(sim_args['sampler']),
            task=SimulatorCopyPhraseTask,
            sampler_args=parse_args(sim_args['sampler_args']))

    if sim_args['verbose']:
        set_verbose(True)
    if task_factory:
        sim_dir = init_simulation_dir(save_location=outdir)
        logger.info(sim_dir)
        task_factory.log_state()

        runner = TaskRunner(save_dir=sim_dir,
                            task_factory=task_factory,
                            runs=runs)
        runner.run()
        metrics.report(sim_dir)


if __name__ == '__main__':
    main()
