"""Task that will replay sessions to compare model predictions on that data.
Used for testing if changes to a model result in more easily differentiated signals."""
import argparse
import logging
from pathlib import Path

from bcipy.io.load import load_json_parameters
from bcipy.simulator.data.sampler.replay_sampler import ReplaySampler
from bcipy.simulator.task.replay_task import ReplayTask
from bcipy.simulator.task.task_factory import TaskFactory
from bcipy.simulator.task.task_runner import TaskRunner
from bcipy.simulator.util.artifact import (DEFAULT_SAVE_LOCATION,
                                           TOP_LEVEL_LOGGER_NAME,
                                           init_simulation_dir)
from bcipy.simulator.util.replay_comparison import comparison_metrics

logger = logging.getLogger(TOP_LEVEL_LOGGER_NAME)


def main():
    """Run the task"""
    parser = argparse.ArgumentParser()
    data_help = ('Raw data folders to be processed.'
                 ' Multiple values can be provided.')
    parser.add_argument("-d",
                        "--data_folder",
                        type=Path,
                        required=False,
                        action='append',
                        help=data_help)
    parser.add_argument("-m",
                        "--model_path",
                        type=Path,
                        required=True,
                        help="Signal model to be used.")
    parser.add_argument("-p",
                        "--parameters",
                        type=Path,
                        required=False,
                        help="Parameter File to be used")
    parser.add_argument("-o",
                        "--output",
                        type=Path,
                        required=False,
                        default=DEFAULT_SAVE_LOCATION,
                        help="Sim output path")
    args = parser.parse_args()
    sim_args = vars(args)

    runs = len(sim_args['data_folder'])
    outdir = sim_args['output']

    parameters = load_json_parameters(sim_args['parameters'], value_cast=True)

    task_factory = TaskFactory(parameters=parameters,
                               source_dirs=sim_args['data_folder'],
                               signal_model_paths=[sim_args['model_path']],
                               sampling_strategy=ReplaySampler,
                               task=ReplayTask)

    sim_dir = init_simulation_dir(save_location=outdir)
    logger.info(sim_dir)
    task_factory.log_state()

    runner = TaskRunner(save_dir=sim_dir, task_factory=task_factory, runs=runs)
    runner.run()

    comparison_metrics(sim_dir, sim_args['data_folder'])


if __name__ == '__main__':
    main()
