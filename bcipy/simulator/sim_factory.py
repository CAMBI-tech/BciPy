""" Factory for Simulator objects """
import logging
from pathlib import Path
from typing import List

from bcipy.helpers.load import load_json_parameters
from bcipy.simulator.helpers.artifact import TOP_LEVEL_LOGGER_NAME
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.metrics import RefereeImpl, SimMetricsHandler
from bcipy.simulator.helpers.model_handler import SigLmModelHandler1
from bcipy.simulator.helpers.sampler import EEGByLetterSampler
from bcipy.simulator.helpers.state_manager import StateManagerImpl
from bcipy.simulator.sim import Simulator, SimulatorCopyPhrase

log = logging.getLogger(TOP_LEVEL_LOGGER_NAME)


class SimulationFactory:
    """ Factory class to create Simulator instances """

    @staticmethod
    def create(source_dirs: List[Path],
               smodel_files: List[str],
               sim_param_path="bcipy/simulator/sim_parameters.json",
               save_dir=None,
               **kwargs) -> Simulator:
        """Create a Simulation, performing all of the necessary setup."""

        # combining parameters
        model_file = Path(smodel_files.pop())
        
        sim_parameters = load_json_parameters(sim_param_path, value_cast=True)
        params_path: str = kwargs.get('parameters', None)
        base_parameters = load_json_parameters(params_path, value_cast=True)
        base_parameters.add_missing_items(sim_parameters)

        data_engine = RawDataEngine(list(map(str, source_dirs)),
                                    base_parameters)
        state_manager = StateManagerImpl(base_parameters)
        sampler = EEGByLetterSampler(data_engine)
        model_handler = SigLmModelHandler1(model_file, base_parameters)


        referee = RefereeImpl(metric_handlers={'basic': SimMetricsHandler()})

        sim = SimulatorCopyPhrase(data_engine,
                                  model_handler,
                                  sampler,
                                  state_manager,
                                  referee,
                                  parameters=base_parameters,
                                  save_dir=save_dir)
        config = {
            'parameters_directory': Path(params_path).parent.name,
            'sampler': str(sampler),
            'model': str(model_handler),
            'task': str(sim)
        }
        for key, value in config.items():
            log.debug(f"{key}: {value}")

        return sim
