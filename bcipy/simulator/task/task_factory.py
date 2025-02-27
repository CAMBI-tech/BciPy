"""Classes and functions for building a simulation."""
import logging
from typing import Any, Dict, List, Optional, Type

from bcipy.core.parameters import DEFAULT_PARAMETERS_PATH, Parameters
from bcipy.helpers.language_model import init_language_model
from bcipy.io.load import load_json_parameters, load_signal_model
from bcipy.signal.model.base_model import SignalModel
from bcipy.simulator.data.data_engine import RawDataEngine
from bcipy.simulator.data.data_process import init_data_processor
from bcipy.simulator.data.sampler import Sampler, TargetNontargetSampler
from bcipy.simulator.task.copy_phrase import SimulatorCopyPhraseTask
from bcipy.simulator.util.artifact import TOP_LEVEL_LOGGER_NAME

logger = logging.getLogger(TOP_LEVEL_LOGGER_NAME)


def update_latest_params(parameters: Parameters) -> None:
    """Update the given parameters with the latest missing values"""
    default_params = load_json_parameters(DEFAULT_PARAMETERS_PATH,
                                          value_cast=True)
    added_params = [
        key for key, change in default_params.diff(parameters).items()
        if change.original_value is None
    ]
    if added_params:
        logger.info(
            f"Added missing parameters using default values: {added_params}")
        parameters.add_missing_items(default_params)


class TaskFactory():
    """Constructs the hierarchy of objects necessary for initializing a task."""

    def __init__(
            self,
            params_path: str,
            source_dirs: List[str],
            signal_model_paths: List[str],
            sampling_strategy: Type[Sampler] = TargetNontargetSampler,
            task: Type[SimulatorCopyPhraseTask] = SimulatorCopyPhraseTask,
            parameters: Optional[Parameters] = None,
            sampler_args: Optional[Dict[str, Any]] = None):

        self.params_path = params_path
        self.signal_model_paths = signal_model_paths

        self.source_dirs = source_dirs
        self.sampling_strategy = sampling_strategy
        self.sampler_args = sampler_args if sampler_args else {}
        self.simulation_task = task

        logger.info("Loading parameters")
        if parameters is None:
            self.parameters = load_json_parameters(self.params_path,
                                                value_cast=True)
        else:
            self.parameters = parameters

        update_latest_params(self.parameters)

        logger.info("Loading signal models")
        self.signal_models = [
            load_signal_model(path) for path in signal_model_paths
        ]

        logger.info("Initializing language model")
        self.language_model = init_language_model(self.parameters)
        self.samplers = self.init_samplers(self.signal_models)

    def log_state(self):
        """Log configured objects of interest. This should be done after the
        sim directory has been created and TOP_LEVEL_LOGGER has been configured,
        which may happen some time after object construction."""
        logger.debug("Language model:")
        logger.debug(f"\t{repr(self.language_model)}")
        logger.debug("Models -> Samplers:")
        logger.debug(f"\t{self.samplers}")

    def init_samplers(
            self,
            signal_models: List[SignalModel]) -> Dict[SignalModel, Sampler]:
        """Initializes the evidence evaluators from the provided signal models.

        Returns a list of evaluators for active devices. Raises an exception if
        more than one evaluator provides the same type of evidence.
        """

        samplers = {}
        for model in signal_models:
            processor = init_data_processor(model)
            logger.info(f"Creating data engine for {model}")
            engine = RawDataEngine(list(map(str, self.source_dirs)),
                                   self.parameters,
                                   data_processor=processor)
            sampler = self.sampling_strategy(engine, **self.sampler_args)
            samplers[model] = sampler
        return samplers

    def make_task(self, run_dir: str) -> SimulatorCopyPhraseTask:
        """Construct a new task"""
        return self.simulation_task(self.parameters,
                                    file_save=run_dir,
                                    signal_models=self.signal_models,
                                    language_model=self.language_model,
                                    samplers=self.samplers)
