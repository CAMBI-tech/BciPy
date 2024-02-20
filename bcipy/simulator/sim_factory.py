""" Factory for Simulator objects """
from pathlib import Path
from typing import List

from bcipy.helpers.load import load_json_parameters
from bcipy.simulator.helpers.data_engine import (RawDataEngine,
                                                 RawDataEngineWrapper)
from bcipy.simulator.helpers.metrics import (MetricReferee, RefereeImpl,
                                             SimMetrics1Handler)
from bcipy.simulator.helpers.model_handler import (ModelHandler,
                                                   SigLmModelHandler1,
                                                   SignalModelHandler1)
from bcipy.simulator.helpers.sampler import EEGByLetterSampler, Sampler
from bcipy.simulator.helpers.state_manager import (StateManager,
                                                   StateManagerImpl)
from bcipy.simulator.sim import SimulatorCopyPhrase


class SimulationFactoryV2:
    """ Factory class to create Simulator instances """

    @staticmethod
    def create(data_folder: str, smodel_files: List[str],
               sim_param_path="bcipy/simulator/sim_parameters.json", save_dir=None, **kwargs):
        # out_dir = kwargs.get('out_dir', Path(__file__).resolve().parent)

        model_file = Path(smodel_files.pop())
        sim_parameters = load_json_parameters(sim_param_path, value_cast=True)
        base_parameters = load_json_parameters(kwargs.get('parameters'), value_cast=True)
        base_parameters.add_missing_items(sim_parameters)

        data_engine = RawDataEngineWrapper(data_folder, base_parameters)
        state_manager: StateManager = StateManagerImpl(base_parameters)
        sampler: Sampler = EEGByLetterSampler(data_engine)
        model_handler: ModelHandler = SigLmModelHandler1(model_file, base_parameters) \
            if sim_parameters.get("sim_lm_active", 0) == 1 else SignalModelHandler1(model_file)
        referee: MetricReferee = RefereeImpl(metric_handlers={'basic': SimMetrics1Handler()})

        sim = SimulatorCopyPhrase(data_engine, model_handler, sampler, state_manager, referee,
                                  parameters=base_parameters, save_dir=save_dir)

        return sim
