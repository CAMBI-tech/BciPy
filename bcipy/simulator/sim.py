import random
from time import sleep
from typing import Optional

import numpy as np

from bcipy.helpers import load, stimuli, symbols
from bcipy.helpers.symbols import alphabet
from bcipy.simulator.helpers.data_engine import DataEngine
from bcipy.simulator.helpers.evidence_fuser import EvidenceFuser
from bcipy.simulator.helpers.sampler import Sampler
from bcipy.simulator.helpers.state_manager import StateManager, SimState
from bcipy.simulator.helpers.types import InquiryResult
from bcipy.simulator.interfaces import MetricReferee, ModelHandler
from bcipy.simulator.simulator_base import Simulator


class SimulatorCopyPhrase(Simulator):
    """

    Copy Phrase simulator.

    Run loop:
        - The loop runs until StateManager determines simulation is over
        - The generated stimuli are passed into sampler, which returns some eeg response
        - The Model handler will generate predictions and feed back into StateManager
        - StateManager will resolve the state while the MetricReferee observes and records any metrics

    Main components:
        - DataEngine:       loading and storing data
        - Sampler:          logic for sampling from data, composed of DataEngine
        - StateManager:     manages run loop details and fuses model evidence with current state
        - MetricReferee:    tracks and scores the simulation on different metrics for later evaluation
        - ModelHandler:     wrapper for models that deals with loading in models and generating evidence

    Requirements:
    - run closed loop simulation of {TaskType} with {data} with {simulation_params}
    """

    def __init__(self, data_engine: DataEngine, model_handler: ModelHandler, sampler: Sampler,
                 state_manager: StateManager, referee: MetricReferee, parameter_path: str = None, save_dir: str = None, verbose=False, visualize=False):
        super(SimulatorCopyPhrase, self).__init__()

        self.save_dir: str = save_dir
        self.model_handler: ModelHandler = model_handler
        self.sampler: Sampler = sampler
        self.referee: MetricReferee = referee
        self.state_manager: StateManager = state_manager

        self.data_engine: DataEngine = data_engine

        self.parameters = self.load_parameters(parameter_path)

        self.symbol_set = alphabet()
        self.write_output = False

        self.verbose = verbose
        self.visualize = visualize

    def run(self):
        while not self.state_manager.is_done():
            print(
                f"Series {self.state_manager.get_state().series_n} | Inquiry {self.state_manager.get_state().inquiry_n} | Target {self.state_manager.get_state().target_symbol}")
            self.state_manager.mutate_state('display_alphabet', self.__make_stimuli(self.state_manager.get_state()))
            sampled_data = self.sampler.sample(self.state_manager.get_state())
            evidence = self.model_handler.generate_evidence(self.state_manager.get_state(),
                                                            sampled_data)  # TODO make this evidence be a dict (mapping of evidence type to evidence)

            print(f"Evidence for stimuli {self.state_manager.get_state().display_alphabet} \n {evidence}")

            inq_record: InquiryResult = self.state_manager.update(evidence)

            print(f"Fused Likelihoods {[str(round(p, 3)) for p in inq_record.fused_likelihood]}")

            if inq_record.decision:
                print(f"Decided {inq_record.decision} for target {inq_record.target} for sentence {self.state_manager.get_state().target_sentence}")

            print("\n")
            sleep(.5)
        # TODO visualize result metrics

    def __make_stimuli(self, state: SimState):
        # TODO abstract out
        subset_length = 10
        val_func: Optional[np.ndarray] = state.get_current_likelihood()
        always_in_stimuli = [symbols.SPACE_CHAR, symbols.BACKSPACE_CHAR]

        if val_func is not None:
            return stimuli.best_selection(self.symbol_set, list(val_func), subset_length, always_included=always_in_stimuli)
        else:
            return random.sample(self.symbol_set, subset_length)

    def load_parameters(self, path):
        # TODO validate parameters
        if not path:
            parameters = self.data_engine.get_parameters()[0]  # TODO fix this parameter logic. for now assuming one parameter file speaks for all
        else:
            parameters = load.load_json_parameters(path, value_cast=True)

        sim_parameters = load.load_json_parameters(
            "bcipy/simulator/sim_parameters.json", value_cast=True)
        parameters.add_missing_items(sim_parameters)
        return parameters

    def get_param(self, name):
        pass
