import dataclasses
import json
import logging
import random
from time import sleep
from typing import Optional, Dict

import numpy as np

from bcipy.helpers import stimuli, symbols
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.symbols import alphabet
from bcipy.simulator.helpers.data_engine import DataEngine
from bcipy.simulator.helpers.metrics import MetricReferee
from bcipy.simulator.helpers.rsvp_utils import format_lm_output
from bcipy.simulator.helpers.sampler import Sampler
from bcipy.simulator.helpers.state_manager import StateManager, SimState
from bcipy.simulator.helpers.types import InquiryResult, SimEvidence
from bcipy.simulator.helpers.log_utils import fmt_stim_likelihoods, fmt_reshaped_evidence
from bcipy.simulator.helpers.model_handler import ModelHandler
from bcipy.simulator.simulator_base import Simulator
from bcipy.task.data import EvidenceType

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
                 state_manager: StateManager, referee: MetricReferee, parameters: Parameters = None,
                 save_dir: str = None):
        super().__init__()

        self.save_dir: str = save_dir
        self.model_handler: ModelHandler = model_handler
        self.sampler: Sampler = sampler
        self.referee: MetricReferee = referee
        self.state_manager: StateManager = state_manager
        self.data_engine: DataEngine = data_engine

        self.parameters: Parameters = self.load_parameters(parameters)

        self.symbol_set = alphabet()
        self.write_output = False

    def run(self):
        log.info(f"SIM START with target word {self.state_manager.get_state().target_sentence}")

        while not self.state_manager.is_done():
            self.state_manager.mutate_state('display_alphabet',
                                            self.__make_stimuli(self.state_manager.get_state()))
            curr_state = self.state_manager.get_state()

            log.info(
                f"Series {curr_state.series_n} | Inquiry {curr_state.inquiry_n} | " +
                f"Target '{curr_state.target_symbol}' | " +
                f"Stimuli {curr_state.display_alphabet}")

            sampled_data = self.sampler.sample(self.state_manager.get_state())
            evidence: Dict[str, SimEvidence] = self.model_handler.generate_evidence(curr_state,
                                                                                    sampled_data)

            log.debug(
                f"EEG Evidence for stimuli {curr_state.display_alphabet} " +
                f"\n {fmt_stim_likelihoods(evidence['sm'].evidence, self.symbol_set)}")

            reshaped_evidence: Dict[str, SimEvidence] = self.__reshape_evidences(evidence)
            log.debug(f"Evidence Shapes {fmt_reshaped_evidence(reshaped_evidence)}")

            inq_record: InquiryResult = self.state_manager.update(reshaped_evidence)
            updated_state = self.state_manager.get_state()

            if inq_record.decision:
                log.info(f"Decided {inq_record.decision} for target {inq_record.target}")

            log.info(f"Current typed: {updated_state.current_sentence}")
            sleep(self.parameters.get("sim_sleep_time", 0.5))

        final_state = self.state_manager.get_state()
        log.info("SIM END")
        log.info(f"FINAL TYPED: {final_state.current_sentence}")
        log.info(self.referee.score(self).__dict__)

        log.debug(f"Final State: {final_state}")

        self.save_run()

    def __make_stimuli(self, state: SimState):
        # TODO abstract out
        subset_length = 10
        val_func: Optional[np.ndarray] = state.get_current_likelihood()
        always_in_stimuli = [symbols.SPACE_CHAR, symbols.BACKSPACE_CHAR]

        if val_func is not None:
            return stimuli.best_selection(self.symbol_set, list(
                val_func), subset_length, always_included=always_in_stimuli)
        elif self.parameters.get('sim_lm_active', 0) == 1:  # use lang model for priors to make stim
            lm_model = self.model_handler.get_model('lm')
            lm_model_evidence = lm_model.predict(list(state.current_sentence))
            val_func = format_lm_output(lm_model_evidence, self.symbol_set)
            return stimuli.best_selection(self.symbol_set, list(
                val_func), subset_length, always_included=always_in_stimuli)

        return random.sample(self.symbol_set, subset_length)

    def __reshape_evidences(self, evidences: Dict[str, SimEvidence]) -> Dict[str, SimEvidence]:

        # reshaping lm_evidence to look like signal model evidence
        reshaped_evidence = evidences.copy()
        if "lm" in evidences:
            unshaped_lm_evidence = evidences["lm"].evidence  # [('B', 0.5), ('C', 0.2), ('_', 0.02)]
            reshaped_lm_lik = format_lm_output(list(unshaped_lm_evidence), self.symbol_set)
            reshaped_evidence['lm'] = SimEvidence(EvidenceType.LM, np.array(reshaped_lm_lik),
                                                  evidences["lm"].symbol_set)

        return reshaped_evidence

    def load_parameters(self, params: Optional[Parameters]):
        # TODO validate parameters
        if params:
            return params
        else:
            return self.data_engine.get_parameters()

    def save_run(self):
        """ Outputs the results of a run to json file """

        # creating result.json object with final state and metrics
        final_state = self.state_manager.get_state()
        final_state_json: Dict = final_state.to_json()
        metric_dict = dataclasses.asdict(self.referee.score(self))
        metric_dict.update(final_state_json)  # adding state data to metrics

        # writing result.json
        with open(f"{self.save_dir}/result.json", 'w') as output_file:
            json.dump(metric_dict, output_file, indent=1)

        # writing params
        self.parameters.save(directory=self.save_dir)

    def get_parameters(self):
        return self.parameters.copy()
