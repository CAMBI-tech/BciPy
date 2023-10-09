import copy
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from bcipy.helpers.exceptions import FieldException
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.symbols import alphabet
from bcipy.simulator.helpers.decision import SimDecisionCriteria, MaxIterationsSim, ProbThresholdSim
from bcipy.simulator.helpers.types import InquiryResult
from bcipy.task.control.criteria import DecisionCriteria, MaxIterationsCriteria, ProbThresholdCriteria


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


class StateManager(ABC):

    def update(self, evidence: np.ndarray):
        raise NotImplementedError()

    def is_done(self) -> bool:
        raise NotImplementedError()

    def get_state(self) -> SimState:
        ...

    def mutate_state(self, state_field, state_value):
        pass


class StateManagerImpl(StateManager):

    def __init__(self, parameters: Parameters):
        self.state: SimState = self.initial_state()
        self.parameters = parameters

        self.stop_inq = 50  # TODO pull from parameters

    def is_done(self) -> bool:

        return self.state.total_inquiry_count() > self.stop_inq or self.state.target_sentence == self.state.current_sentence or self.state.series_n > 50

    def update(self, evidence: np.ndarray) -> InquiryResult:

        temp_inquiry_result = InquiryResult(target=self.state.target_symbol, time_spent=0, stimuli=self.state.display_alphabet,
                                            evidence_likelihoods=list(evidence), decision=None)

        # finding out whether max iterations is hit or prob threshold is hit
        temp_series = copy.deepcopy(self.get_state().series_results)
        temp_series[-1].append(temp_inquiry_result)
        is_decidable = any([decider.decide(temp_series[-1]) for decider in self.state.decision_criterion])
        decision = None
        # TODO what to do when max inquiry count is reached?

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
                                           evidence_likelihoods=list(evidence), decision=decision)

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
        target_symbol = sentence[0]
        default_criterion: List[SimDecisionCriteria] = [MaxIterationsSim(50), ProbThresholdSim(0.8)]

        return SimState(target_symbol=target_symbol, current_sentence="", target_sentence=sentence, display_alphabet=[], inquiry_n=0, series_n=0,
                        series_results=[[]], decision_criterion=default_criterion)
