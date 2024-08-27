import dataclasses
import json
import logging
import random
from time import sleep
from typing import Dict, Optional

import numpy as np

from bcipy.helpers import stimuli, symbols
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.session import session_excel
from bcipy.helpers.symbols import alphabet
from bcipy.simulator.helpers.data_engine import DataEngine
from bcipy.simulator.helpers.log_utils import (fmt_reshaped_evidence,
                                               fmt_stim_likelihoods)
from bcipy.simulator.helpers.metrics import MetricReferee
from bcipy.simulator.helpers.model_handler import ModelHandler
from bcipy.simulator.helpers.rsvp_utils import format_lm_output
from bcipy.simulator.helpers.sampler import Sampler
from bcipy.simulator.helpers.state_manager import (SimState, StateManager,
                                                   format_sim_state_dump)
from bcipy.simulator.helpers.types import InquiryResult, SimEvidence
from bcipy.simulator.simulator_base import Simulator
from bcipy.task.data import EvidenceType, Inquiry, Session

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
                 state_manager: StateManager, referee: MetricReferee, parameters: Optional[Parameters] = None,
                 save_dir: Optional[str] = None):
        super().__init__()

        self.save_dir = save_dir
        self.model_handler = model_handler
        self.sampler = sampler
        self.referee = referee
        self.state_manager = state_manager
        self.data_engine = data_engine

        self.parameters = self.load_parameters(parameters)

        self.symbol_set = alphabet()
        self.write_output = False

    def run(self):
        assert self.save_dir, "Save directory not set"
        log.info(f"SIM START with target word {self.state_manager.get_state().target_sentence}")

        session = Session(
            save_location=self.save_dir,
            task='Copy Phrase Simulation',
            mode='RSVP',
            symbol_set=self.symbol_set,
            decision_threshold=self.parameters['decision_threshold'])

        while not self.state_manager.is_done():
            self.state_manager.mutate_state('display_alphabet',
                                            self.__make_stimuli(self.state_manager.get_state()))
            curr_state = self.state_manager.get_state()

            log.info(
                f"Series {curr_state.series_n} | Inquiry {curr_state.inquiry_n} | " +
                f"Target '{curr_state.target_symbol}' | " +
                f"Stimuli {curr_state.display_alphabet}")

            sampled_data = self.sampler.sample_data(self.state_manager.get_state())
            evidence: Dict[str, SimEvidence] = self.model_handler.generate_evidence(curr_state,
                                                                                    sampled_data)

            log.debug(
                f"EEG Evidence for stimuli {curr_state.display_alphabet} " +
                f"\n {fmt_stim_likelihoods(evidence['sm'].evidence, self.symbol_set)}")

            reshaped_evidence: Dict[str, SimEvidence] = self.__reshape_evidences(evidence)
            log.debug(f"Evidence Shapes {fmt_reshaped_evidence(reshaped_evidence)}")

            inq_record: InquiryResult = self.state_manager.update(reshaped_evidence)
            updated_state = self.state_manager.get_state()

            data = self.new_data_record(curr_state, inq_record)
            session.add_sequence(data)

            if inq_record.decision:
                log.info(f"Decided {inq_record.decision} for target {inq_record.target}")
                session.add_series()

            log.info(f"Current typed: {updated_state.current_sentence}")
            sleep(self.parameters.get("sim_sleep_time", 0.5))

        final_state = self.state_manager.get_state()
        log.info("SIM END")
        log.info(f"FINAL TYPED: {final_state.current_sentence}")
        log.info(self.referee.score(self).__dict__)

        self.save_run(session)

    def has_language_model(self) -> bool:
        """Indicates whether this task has been configured with a language model"""
        return self.model_handler.includes_language_model()

    def __make_stimuli(self, state: SimState):
        """
        Creates next set of stimuli:
            - uses current likelihoods while during a series
            - uses language model for beginning of series
            - random letters in case of no LM and beginning of series
        """
        subset_length = self.parameters['stim_length']
        priors: Optional[np.ndarray] = state.get_current_likelihood()
        always_in_stimuli = []
        if self.parameters.get('backspace_always_shown', False):
            always_in_stimuli.append(symbols.BACKSPACE_CHAR)

        # during series
        if priors is not None:
            return stimuli.best_selection(self.symbol_set, list(
                priors), subset_length, always_included=always_in_stimuli)
        # beginning of series
        elif self.has_language_model():
            # use lang model for priors to make stim
            lm_model = self.model_handler.get_model('lm')
            lm_model_evidence = lm_model.predict(list(state.current_sentence))
            priors = format_lm_output(lm_model_evidence, self.symbol_set)
            return stimuli.best_selection(self.symbol_set, list(
                priors), subset_length, always_included=always_in_stimuli)

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

    def new_data_record(self, inquiry_state: SimState, inquiry_result: InquiryResult) -> Inquiry:
        """Construct a new inquiry data record.

        Parameters
        ----------
        - inquiry_state : state prior to presenting the inquiry
        - inquiry_result : result of presenting the inquiry
        - post_inquiry_state : updated state

        Returns
        -------
        Inquiry data for the current schedule
        """
        target_info = [
            'target' if inquiry_result == sym else 'nontarget'
            for sym in inquiry_result.stimuli
        ]

        data = Inquiry(stimuli=inquiry_result.stimuli,
                       timing=[],
                       triggers=[],
                       target_info=target_info,
                       target_letter=inquiry_result.target,
                       current_text=inquiry_state.current_sentence,
                       target_text=inquiry_state.target_sentence)
        data.precision = 4
        data.evidences = {
            val.evidence_type: list(val.evidence)
            for val in inquiry_result.evidences.values()
        }
        if EvidenceType.LM not in data.evidences:
            data.evidences[EvidenceType.LM] = []
        data.likelihood = list(inquiry_result.fused_likelihood)
        return data

    def save_run(self, session: Session):
        """ Outputs the results of a run to json file """
        assert self.save_dir, "Save directory not set"
        # creating result.json object with final state and metrics
        final_state = self.state_manager.get_state()
        final_state_json: Dict = format_sim_state_dump(final_state)
        metric_dict = dataclasses.asdict(self.referee.score(self))
        metric_dict.update(final_state_json)  # adding state data to metrics

        # writing result.json
        with open(f"{self.save_dir}/result.json", 'w',
                  encoding='utf8') as output_file:
            json.dump(metric_dict, output_file, indent=2)

        with open(f"{self.save_dir}/session.json", 'wt',
                  encoding='utf8') as session_file:
            json.dump(session.as_dict(), session_file, indent=2)

        if session.has_evidence():
            session_excel(session=session,
                          excel_file=f"{self.save_dir}/session.xlsx")

    def reset(self):
        self.state_manager.reset_state()

    def get_parameters(self):
        return self.parameters.copy()

    def __str__(self):
        return f"<{self.__class__.__name__}>"
