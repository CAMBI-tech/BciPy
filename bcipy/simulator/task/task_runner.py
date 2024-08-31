"""Setup and run the task"""

import argparse
import logging
import os
from glob import glob
from pathlib import Path
from typing import List, Type

from bcipy.helpers.language_model import init_language_model
from bcipy.helpers.load import load_json_parameters, load_signal_models
from bcipy.signal.model.base_model import SignalModel
from bcipy.simulator.helpers import artifact
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.data_process import init_data_processor
from bcipy.simulator.helpers.sampler import EEGByLetterSampler, Sampler
from bcipy.simulator.task.copy_phrase import SimTask, SimulatorCopyPhraseTask

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


class TaskFactory():
    """Constructs the hierarchy of objects necessary for initializing a task."""

    def __init__(self,
                 params_path: str,
                 model_path: str,
                 source_dirs: List[str],
                 sampling_strategy: Type[Sampler] = EEGByLetterSampler,
                 task: Type[SimTask] = SimulatorCopyPhraseTask):
        self.params_path = params_path
        self.model_path = model_path
        self.source_dirs = source_dirs
        self.sampling_strategy = sampling_strategy
        self.simulation_task = task

        logger.info("Loading parameters")
        self.parameters = load_json_parameters(self.params_path,
                                               value_cast=True)

        logger.info("Loading signal models")
        self.signal_models = load_signal_models(directory=self.model_path)
        logger.debug(self.signal_models)

        logger.info("Initializing language model")
        self.language_model = init_language_model(self.parameters)
        logger.debug(self.language_model)

        self.samplers = self.init_samplers(self.signal_models)
        logger.debug(self.samplers)

    def init_samplers(self, signal_models: List[SignalModel]) -> List[Sampler]:
        """Initializes the evidence evaluators from the provided signal models.

        Returns a list of evaluators for active devices. Raises an exception if
        more than one evaluator provides the same type of evidence.
        """

        samplers = []
        for model in signal_models:
            processor = init_data_processor(model)
            logger.info(f"Creating data engine for {model}")
            engine = RawDataEngine(list(map(str, self.source_dirs)),
                                   self.parameters,
                                   data_processor=processor)
            sampler = self.sampling_strategy(engine)
            samplers.append(sampler)
        return samplers

    def make_task(self, run_dir: str) -> SimTask:
        """Construct a new task"""
        return self.simulation_task(self.parameters,
                                    file_save=run_dir,
                                    signal_models=self.signal_models,
                                    language_model=self.language_model,
                                    samplers=self.samplers)


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
    task_factory = TaskFactory(params_path=sim_args['parameters'],
                               model_path=sim_args['model_path'],
                               source_dirs=sim_args['source_dirs'],
                               sampling_strategy=EEGByLetterSampler,
                               task=SimulatorCopyPhraseTask)
    runner = TaskRunner(save_dir=sim_dir,
                        task_factory=task_factory,
                        runs=sim_args['n'])
    runner.run()


if __name__ == '__main__':
    main()
