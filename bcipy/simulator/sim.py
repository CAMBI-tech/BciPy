import logging
import random
from time import sleep
from typing import Optional

import numpy as np

from bcipy.helpers import load, stimuli, symbols
from bcipy.helpers.symbols import alphabet
from bcipy.simulator.helpers.data_engine import DataEngine
from bcipy.simulator.helpers.metrics import MetricReferee
from bcipy.simulator.helpers.sampler import Sampler
from bcipy.simulator.helpers.state_manager import StateManager, SimState
from bcipy.simulator.helpers.types import InquiryResult
from bcipy.simulator.helpers.log_utils import format_alp_likelihoods
from bcipy.simulator.helpers.model_handler import ModelHandler
from bcipy.simulator.simulator_base import Simulator

log = logging.getLogger(__name__)


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
        log.info(f"SIM START with target word {self.state_manager.get_state().target_sentence}")

        while not self.state_manager.is_done():
            curr_state = self.state_manager.get_state()
            log.info(f"Series {curr_state.series_n} | Inquiry {curr_state.inquiry_n} | Target {curr_state.target_symbol}")

            self.state_manager.mutate_state('display_alphabet', self.__make_stimuli(curr_state))
            sampled_data = self.sampler.sample(curr_state)
            evidence = self.model_handler.generate_evidence(curr_state,
                                                            sampled_data)  # TODO make this evidence be a dict (mapping of evidence type to evidence)

            log.debug(f"Evidence for stimuli {curr_state.display_alphabet} \n {format_alp_likelihoods(evidence, self.symbol_set)}")

            inq_record: InquiryResult = self.state_manager.update(evidence)
            updated_state = self.state_manager.get_state()
            log.debug(f"Fused Likelihoods {format_alp_likelihoods(inq_record.fused_likelihood, self.symbol_set)}")

            if inq_record.decision:
                log.info(f"Decided {inq_record.decision} for target {inq_record.target}")

            log.info(f"Current typed: {updated_state.current_sentence}")
            sleep(self.parameters.get("sim_sleep_time", 0.5))

        final_state = self.state_manager.get_state()
        log.info(f"SIM END")
        log.info(f"FINAL TYPED: {final_state.current_sentence}")
        log.info(self.referee.score(self).__dict__)

        log.debug(f"Final State: {final_state}")


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
