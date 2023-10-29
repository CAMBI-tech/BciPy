import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

from bcipy.helpers.exceptions import FieldException
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.symbols import alphabet
from bcipy.simulator.helpers.decision import SimDecisionCriteria, MaxIterationsSim, ProbThresholdSim
from bcipy.simulator.helpers.evidence_fuser import MultiplyFuser, EvidenceFuser
from bcipy.simulator.helpers.types import InquiryResult
from bcipy.task.control.criteria import DecisionCriteria, MaxIterationsCriteria, ProbThresholdCriteria
from bcipy.task.control.handler import EvidenceFusion


@dataclass
class SimState:
    """ Represents the state of a current session during simulation """
    target_symbol: str
    current_sentence: str
    target_sentence: str
    display_alphabet: List[str]
    inquiry_n: int
    series_n: int
    series_results: List[List[InquiryResult]]
    decision_criterion: List[SimDecisionCriteria]

    def total_inquiry_count(self):
        count = 0
        for series in self.series_results:
            count += len(series)

        return count

    def get_current_likelihood(self) -> Optional[np.ndarray]:
        cur_likelihood = None
        if self.inquiry_n:
            cur_likelihood = self.series_results[-1][-1].fused_likelihood

        return cur_likelihood


class StateManager(ABC):

    def update(self, evidence: np.ndarray):  # TODO change evidence type to dictionary or some dataclass
        raise NotImplementedError()

    def is_done(self) -> bool:
        raise NotImplementedError()

    def get_state(self) -> SimState:
        ...

    def mutate_state(self, state_field, state_value):
        pass


class StateManagerImpl(StateManager):

    def __init__(self, parameters: Parameters, fuser_class=MultiplyFuser):
        self.state: SimState = self.initial_state()
        self.parameters = parameters
        self.fuser_class: EvidenceFuser.__class__ = fuser_class

        self.max_inq_len = self.parameters.get('max_inq_len', 50)

    def is_done(self) -> bool:
        # TODO add stoppage criterion, Stoppage criterion is seperate from decision. Decision should we go on to next letter or not
        return self.state.total_inquiry_count() > self.max_inq_len or self.state.target_sentence == self.state.current_sentence or self.state.series_n > 50

    def update(self, evidence) -> InquiryResult:

        fuser = self.fuser_class()
        current_series: List[InquiryResult] = self.state.series_results[self.state.series_n]
        prior_likelihood: Optional[np.ndarray] = current_series.pop().fused_likelihood if current_series else None  # most recent likelihood
        evidence_dict = {"SM": evidence}  # TODO create wrapper object for Evidences
        fused_likelihood = fuser.fuse(prior_likelihood, evidence_dict)

        # finding out whether max iterations is hit or prob threshold is hit
        temp_inquiry_result = InquiryResult(target=self.state.target_symbol, time_spent=0, stimuli=self.state.display_alphabet,
                                            evidence_likelihoods=list(evidence), fused_likelihood=fused_likelihood,  # TODO change to use evidence_dict
                                            decision=None)

        temp_series = copy.deepcopy(self.get_state().series_results)
        temp_series[-1].append(temp_inquiry_result)
        is_decidable = any([decider.decide(temp_series[-1]) for decider in self.state.decision_criterion])
        decision = None

        new_state = self.get_state().__dict__
        if is_decidable:
            decision = alphabet()[np.argmax(evidence)]  # deciding the maximum probability symbol TODO abstract
            if decision == self.state.target_symbol:  # correct decision
                new_state['series_n'] += 1  # TODO abstract out into reset function
                new_state['series_results'].append([])
                new_state['inquiry_n'] = 0

                new_state['current_sentence'] += decision

                next_target_symbol_idx = len(new_state['current_sentence'])
                if next_target_symbol_idx >= len(self.state.target_sentence):
                    pass
                else:
                    new_state['target_symbol'] = self.state.target_sentence[next_target_symbol_idx]

            else:  # wrong decision
                new_state['series_n'] += 1
                new_state['series_results'].append([])
                new_state['inquiry_n'] = 0

        else:
            new_state['inquiry_n'] += 1

        new_inquiry_result = InquiryResult(target=self.state.target_symbol, time_spent=0, stimuli=self.state.display_alphabet,
                                           evidence_likelihoods=list(evidence), decision=decision, fused_likelihood=fused_likelihood)

        new_state['series_results'][self.state.series_n].append(new_inquiry_result)

        self.state = SimState(**new_state)

        return new_inquiry_result

    def get_state(self) -> SimState:
        return copy.copy(self.state)

    def mutate_state(self, state_field, state_value) -> SimState:
        state_dict = self.get_state().__dict__
        if state_field in state_dict:
            state_dict[state_field] = state_value
            self.state = SimState(**state_dict)
            return self.get_state()
        else:
            raise FieldException(f"Cannot find state field {state_field}")

    @staticmethod
    def initial_state(parameters: Parameters = None) -> SimState:
        sentence = "HELLO_WORLD"  # TODO abstract out with sim_parameters.json
        target_symbol = sentence[0]  # TODO use parameters.get('spelled_letters_count')
        default_criterion: List[SimDecisionCriteria] = [MaxIterationsSim(50), ProbThresholdSim(0.8)]

        evidence_types = parameters.get(
            'evidence_types') if parameters else None  # TODO make new parameter and create default series_likelihoods object based off that

        return SimState(target_symbol=target_symbol, current_sentence="", target_sentence=sentence, display_alphabet=[], inquiry_n=0, series_n=0,
                        series_results=[[]], decision_criterion=default_criterion)
