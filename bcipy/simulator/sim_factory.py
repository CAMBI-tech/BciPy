""" Factory for Simulator objects """
from pathlib import Path
from typing import List

from bcipy.helpers.load import load_json_parameters
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.metrics import (MetricReferee, RefereeImpl,
                                             SimMetricsHandler)
from bcipy.simulator.helpers.model_handler import (ModelHandler,
                                                   SigLmModelHandler1)
from bcipy.simulator.helpers.sampler import EEGByLetterSampler, Sampler
from bcipy.simulator.helpers.state_manager import (StateManager,
                                                   StateManagerImpl)
from bcipy.simulator.sim import Simulator, SimulatorCopyPhrase


class SimulationFactory:
    """ Factory class to create Simulator instances """

    @staticmethod
    def create(source_dirs: List[Path],
               smodel_files: List[str],
               sim_param_path="bcipy/simulator/sim_parameters.json",
               save_dir=None,
               **kwargs) -> Simulator:
        """Create a Simulation, performing all of the necessary setup."""
        # out_dir = kwargs.get('out_dir', Path(__file__).resolve().parent)

        # combining parameters
        model_file = Path(smodel_files.pop())
        sim_parameters = load_json_parameters(sim_param_path, value_cast=True)
        params_path: str = kwargs.get('parameters', None)
        base_parameters = load_json_parameters(params_path, value_cast=True)
        base_parameters.add_missing_items(sim_parameters)

        data_engine = RawDataEngine(list(map(str, source_dirs)),
                                    base_parameters)
        state_manager: StateManager = StateManagerImpl(base_parameters)
        sampler: Sampler = EEGByLetterSampler(data_engine)
        model_handler: ModelHandler = SigLmModelHandler1(
            model_file, base_parameters)
        referee: MetricReferee = RefereeImpl(
            metric_handlers={'basic': SimMetricsHandler()})

        sim = SimulatorCopyPhrase(data_engine,
                                  model_handler,
                                  sampler,
                                  state_manager,
                                  referee,
                                  parameters=base_parameters,
                                  save_dir=save_dir)

        return sim