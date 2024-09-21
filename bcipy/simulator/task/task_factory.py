import logging
from typing import Dict, List, Type

from bcipy.helpers.language_model import init_language_model
from bcipy.helpers.load import load_json_parameters, load_signal_models
from bcipy.signal.model.base_model import SignalModel
from bcipy.simulator.helpers import artifact
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.data_process import init_data_processor
from bcipy.simulator.helpers.sampler import Sampler, TargetNontargetSampler
from bcipy.simulator.task.copy_phrase import SimulatorCopyPhraseTask

logger = logging.getLogger(artifact.TOP_LEVEL_LOGGER_NAME)


class TaskFactory():
    """Constructs the hierarchy of objects necessary for initializing a task."""

    # TODO: sampling strategy per model type
    def __init__(self,
                 params_path: str,
                 model_path: str,
                 source_dirs: List[str],
                 sampling_strategy: Type[Sampler] = TargetNontargetSampler,
                 task: Type[SimulatorCopyPhraseTask] = SimulatorCopyPhraseTask):
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

    def init_samplers(self, signal_models: List[SignalModel]) -> Dict[SignalModel, Sampler]:
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
            sampler = self.sampling_strategy(engine)
            samplers[model] = sampler
        return samplers

    def make_task(self, run_dir: str) -> SimulatorCopyPhraseTask:
        """Construct a new task"""
        return self.simulation_task(self.parameters,
                                    file_save=run_dir,
                                    signal_models=self.signal_models,
                                    language_model=self.language_model,
                                    samplers=self.samplers)
