"""Setup and run the task"""

import argparse
import logging
import os
from glob import glob
from pathlib import Path

from bcipy.simulator.helpers import artifact
from bcipy.simulator.helpers.sampler import TargetNontargetSampler
from bcipy.simulator.task.copy_phrase import SimulatorCopyPhraseTask
from bcipy.simulator.task.task_factory import TaskFactory

logger = logging.getLogger(artifact.TOP_LEVEL_LOGGER_NAME)


def configure_run_directory(sim_dir: str, run: int) -> str:
    """Create the necessary directories and configure the logger.
    Returns the run directory.
    """
    run_name = f"run_{run}"
    path = f"{sim_dir}/{run_name}"
    os.mkdir(path)
    artifact.configure_logger(log_path=path,
                              file_name=f"{run_name}.log",
                              use_stdout=False)
    return path


class TaskRunner():
    """Responsible for executing a task a given number of times."""

    def __init__(self,
                 save_dir: str,
                 task_factory: TaskFactory,
                 runs: int = 1):

        self.task_factory = task_factory
        self.sim_dir = save_dir
        self.runs = runs

    def run(self):
        """Run one or more simulations"""
        self.task_factory.parameters.save(self.sim_dir)
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
                 ' Ex. "*RSVP_Copy_Phrase*"')
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data_folder",
                        type=Path,
                        required=True,
                        help="Raw data folders to be processed.")
    parser.add_argument('-g', '--glob_pattern', help=glob_help, default="*")
    parser.add_argument("-m",
                        "--model_path",
                        type=Path,
                        required=True,
                        help="Signal models to be used")
    parser.add_argument("-p",
                        "--parameters",
                        type=Path,
                        required=True,
                        help="Parameter File to be used")
    parser.add_argument("-n",
                        type=int,
                        required=False,
                        default=1,
                        help="Number of times to run the simulation")
    args = parser.parse_args()
    sim_args = vars(args)

    sim_args['source_dirs'] = [
        Path(d) for d in glob(str(Path(args.data_folder, args.glob_pattern)))
        if Path(d).is_dir()
    ]

    sim_dir = artifact.init_simulation_dir()
    logger.info(sim_dir)
    task_factory = TaskFactory(params_path=sim_args['parameters'],
                               model_path=sim_args['model_path'],
                               source_dirs=sim_args['source_dirs'],
                               sampling_strategy=TargetNontargetSampler,
                               task=SimulatorCopyPhraseTask)
    runner = TaskRunner(save_dir=sim_dir,
                        task_factory=task_factory,
                        runs=sim_args['n'])
    runner.run()


if __name__ == '__main__':
    main()
