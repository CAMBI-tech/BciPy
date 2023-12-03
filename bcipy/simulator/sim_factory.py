from pathlib import Path
from typing import List

import numpy as np

from bcipy.config import DEFAULT_PARAMETER_FILENAME
from bcipy.helpers.load import load_json_parameters
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.metrics import MetricReferee, DummyRef
from bcipy.simulator.helpers.model_handler import SignalModelHandler1, ModelHandler
from bcipy.simulator.helpers.sampler import Sampler, EEGByLetterSampler
from bcipy.simulator.helpers.state_manager import StateManager, StateManagerImpl
from bcipy.simulator.sim import SimulatorCopyPhrase
from bcipy.simulator.sim_copy_phrase import SimulatorCopyPhraseReplay
from bcipy.simulator.simulator_base import Simulator


class SimulationFactory:

    @staticmethod
    def create(
            sim_task="",
            parameter_path="",
            smodel_files=None,
            lmodel_files=None,
            data_folders=None,
            out_dir="",
            **kwargs) -> Simulator:
        if sim_task == 'RSVP_COPY_PHRASE':
            # TODO validate arguments

            if not parameter_path:
                data_folder = data_folders[0]
                parameter_path = Path(data_folder, DEFAULT_PARAMETER_FILENAME)

            return SimulatorCopyPhraseReplay(parameter_path, out_dir, smodel_files, lmodel_files, data_folders[0],
                                             verbose=kwargs.get('verbose', False))

        # TODO refactor for sampling simulator


class SimulationFactoryV2:
    @staticmethod
    def create(data_folders: List[str], smodel_files: List[str], sim_param_path="bcipy/simulator/sim_parameters.json", **kwargs):
        out_dir = kwargs.get('out_dir', Path(__file__).resolve().parent)

        model_file = Path(smodel_files.pop())
        sim_parameters = load_json_parameters(sim_param_path, value_cast=True)

        data_engine = RawDataEngine(data_folders)
        stateManager: StateManager = StateManagerImpl(sim_parameters)
        sampler: Sampler = EEGByLetterSampler(data_engine)
        model_handler: ModelHandler = SignalModelHandler1(model_file)
        referee: MetricReferee = DummyRef()

        sim = SimulatorCopyPhrase(data_engine, model_handler, sampler, stateManager, referee)

        return sim
