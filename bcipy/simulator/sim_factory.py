from pathlib import Path
from typing import List

from bcipy.helpers.load import load_json_parameters
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.metrics import MetricReferee, RefereeImpl, SimMetrics1Handler
from bcipy.simulator.helpers.model_handler import SignalModelHandler1, ModelHandler
from bcipy.simulator.helpers.sampler import Sampler, EEGByLetterSampler
from bcipy.simulator.helpers.state_manager import StateManager, StateManagerImpl
from bcipy.simulator.sim import SimulatorCopyPhrase


class SimulationFactoryV2:
    @staticmethod
    def create(data_folders: List[str], smodel_files: List[str],
               sim_param_path="bcipy/simulator/sim_parameters.json", **kwargs):
        out_dir = kwargs.get('out_dir', Path(__file__).resolve().parent)

        model_file = Path(smodel_files.pop())
        sim_parameters = load_json_parameters(sim_param_path, value_cast=True)

        data_engine = RawDataEngine(data_folders)
        stateManager: StateManager = StateManagerImpl(sim_parameters)
        sampler: Sampler = EEGByLetterSampler(data_engine)
        model_handler: ModelHandler = SignalModelHandler1(model_file)
        referee: MetricReferee = RefereeImpl(metric_handlers={'basic': SimMetrics1Handler()})

        sim = SimulatorCopyPhrase(data_engine, model_handler, sampler, stateManager, referee)

        return sim
