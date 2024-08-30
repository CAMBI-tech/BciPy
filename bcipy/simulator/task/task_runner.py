"""Setup and run the task"""

import argparse
import logging
import os
from glob import glob
from pathlib import Path
from typing import List

from bcipy.helpers.language_model import init_language_model
from bcipy.helpers.load import load_json_parameters, load_signal_models
from bcipy.simulator.helpers import artifact
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.data_process import EegRawDataProcessor
from bcipy.simulator.helpers.sampler import EEGByLetterSampler
from bcipy.simulator.task.copy_phrase import SimulatorCopyPhraseTask
from bcipy.task.main import Task

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
                 model_path: str,
                 params_path: str,
                 source_dirs: List[str],
                 save_dir: str,
                 runs: int = 1):
        self.model_path = model_path
        self.params_path = params_path
        self.source_dirs = source_dirs
        self.sim_dir = save_dir
        self.runs = runs

        self.sampling_strategy = EEGByLetterSampler
        self.simulation_task = SimulatorCopyPhraseTask

        self.parameters = None
        self.signal_models = []
        self.language_model = None
        self.sampler = None

    def setup(self) -> None:
        """Setup the task objects"""
        logger.info("Loading parameters")
        self.parameters = load_json_parameters(self.params_path,
                                               value_cast=True)

        logger.info("Loading signal models")
        self.signal_models = load_signal_models(directory=self.model_path)
        logger.debug(self.signal_models)

        logger.info("Initializing language model")
        self.language_model = init_language_model(self.parameters)
        logger.debug(self.language_model)

        logger.info("Creating data engine")
        # TODO: initialize a data engine and sampler for each model
        data_processor = EegRawDataProcessor(self.signal_models[0])
        data_engine = RawDataEngine(list(map(str, self.source_dirs)),
                                    self.parameters, data_processor)
        self.sampler = self.sampling_strategy(data_engine)
        logger.debug("Using sampler:")
        logger.debug(self.sampler)

    def setup_complete(self) -> bool:
        """Determines if the runner has been setup correctly."""
        return all([
            self.parameters is not None,
            len(self.signal_models) > 0, self.language_model is not None,
            self.sampler is not None
        ])

    def make_task(self, run_dir) -> Task:
        """Create the task. This is done for every run."""
        return self.simulation_task(self.parameters,
                                    file_save=run_dir,
                                    signal_models=self.signal_models,
                                    language_model=self.language_model,
                                    sampler=self.sampler)

    def run(self):
        """Run one or more simulations"""
        if not self.setup_complete():
            self.setup()
        self.parameters.save(self.sim_dir)
        for i in range(self.runs):
            logger.info(f"Executing task {i+1}")
            self.do_run(i + 1)
            logger.info("Task complete")

    def do_run(self, run: int):
        """Execute a simulation run."""
        run_dir = configure_run_directory(self.sim_dir, run)
        logger.info(run_dir)
        task = self.make_task(run_dir)
        task.execute()


def main():
    """Run the task"""
    glob_help = ('glob pattern to select a subset of data folders'
                 ' Ex. "*RSVP_Copy_Phrase*"')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_folder",
        type=Path,
        required=True,
        help=
        "Raw data folders to be processed. Singular wrapper dir with data folders"
    )
    parser.add_argument('-g', '--glob_pattern', help=glob_help, default="*")
    parser.add_argument("-m",
                        "--model_path",
                        type=Path,
                        required=True,
                        help="Signal model to be used")
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
    runner = TaskRunner(model_path=sim_args['model_path'],
                        params_path=sim_args['parameters'],
                        source_dirs=sim_args['source_dirs'],
                        save_dir=sim_dir,
                        runs=sim_args['n'])
    runner.run()


if __name__ == '__main__':
    main()
