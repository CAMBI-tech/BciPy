import copy
import dataclasses
import logging
import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

from bcipy.helpers.exceptions import FieldException
from bcipy.helpers.language_model import histogram
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.symbols import alphabet, BACKSPACE_CHAR
from bcipy.simulator.helpers.decision import SimDecisionCriteria, MaxIterationsSim, ProbThresholdSim
from bcipy.simulator.helpers.evidence_fuser import MultiplyFuser, EvidenceFuser
from bcipy.simulator.helpers.log_utils import fmt_likelihoods_for_hist
from bcipy.simulator.helpers.rsvp_utils import next_target_letter
from bcipy.simulator.helpers.types import InquiryResult, SimEvidence

log = logging.getLogger(__name__)


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

    def to_json(self):
        d = dataclasses.asdict(self)
        d['series_results'] = [list(map(lambda ir: ir.to_json(), lis)) for lis in
                               self.series_results]

        d['decision_criterion'] = [str(dec) for dec in self.decision_criterion]

        return d


class StateManager(ABC):

    # TODO change evidence type to dictionary or some dataclass
    def update(self, evidence: Dict[str, SimEvidence]):
        raise NotImplementedError()

    def is_done(self) -> bool:
        raise NotImplementedError()

    def get_state(self) -> SimState:
        ...

    def mutate_state(self, state_field, state_value):
        pass

    def reset_state(self):
        ...


class StateManagerImpl(StateManager):

    def __init__(self, parameters: Parameters, fuser_class=MultiplyFuser):
        self.state: SimState = self.initial_state()
        self.parameters = parameters
        self.fuser_class: EvidenceFuser.__class__ = fuser_class

        self.max_inq_len = self.parameters.get('max_inq_len', 100)

    def is_done(self) -> bool:
        # TODO add stoppage criterion, Stoppage criterion is seperate from
        # decision. Decision should we go on to next letter or not
        return self.state.total_inquiry_count(
        ) > self.max_inq_len or self.state.target_sentence == self.state.current_sentence or self.state.series_n > 50

    def update(self, evidence) -> InquiryResult:
        """ Updating the current state based on provided evidence
            - fuses prior evidence with current evidence and makes potential decision
            - stores inquiry, including typing decision or lack of decision
            - resets series on decision and updates target letter
        """

        fuser = self.fuser_class()
        current_series: List[InquiryResult] = self.state.series_results[self.state.series_n]
        prior_likelihood: Optional[np.ndarray] = current_series.pop(
        ).fused_likelihood if current_series else None  # most recent likelihood

        fused_likelihood = fuser.fuse(prior_likelihood, evidence)

        # finding out whether max iterations is hit or prob threshold is hit
        temp_inquiry_result = InquiryResult(target=self.state.target_symbol, time_spent=0,
                                            stimuli=self.state.display_alphabet,
                                            # TODO change to use evidence_dict
                                            evidences=evidence,
                                            fused_likelihood=fused_likelihood,
                                            decision=None)

        # can we make decision
        temp_series = copy.deepcopy(self.get_state().series_results)
        temp_series[-1].append(temp_inquiry_result)
        is_decidable = any(
            [decider.decide(temp_series[-1]) for decider in self.state.decision_criterion])
        decision = None

        new_state = self.get_state().__dict__
        new_inquiry_result = InquiryResult(target=self.state.target_symbol, time_spent=0,
                                           stimuli=self.state.display_alphabet,
                                           evidences=evidence, decision=decision,
                                           fused_likelihood=fused_likelihood)

        log.debug(
            f"Fused Likelihoods | current typed - {self.state.current_sentence} | stimuli {self.state.display_alphabet} \n "
            f"{histogram(fmt_likelihoods_for_hist(new_inquiry_result.fused_likelihood, alphabet()))}")

        new_state['series_results'][self.state.series_n].append(new_inquiry_result)
        if is_decidable:
            decision = alphabet()[
                np.argmax(
                    fused_likelihood)]  # deciding the maximum probability symbol TODO abstract

            # resetting series
            new_state['series_n'] += 1
            new_state['series_results'].append([])
            new_state['inquiry_n'] = 0

            # updating current sentence and finding next target
            new_state['current_sentence'] = new_state['current_sentence'] + decision \
                if decision != BACKSPACE_CHAR else new_state['current_sentence'][:-1]
            new_state['target_symbol'] = next_target_letter(new_state['current_sentence'],
                                                            self.state.target_sentence)

        else:
            new_state['inquiry_n'] += 1

        new_inquiry_result = InquiryResult(target=self.state.target_symbol, time_spent=0,
                                           stimuli=self.state.display_alphabet,
                                           evidences=evidence, decision=decision,
                                           fused_likelihood=fused_likelihood)

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

        raise FieldException(f"Cannot find state field {state_field}")

    def reset_state(self):
        self.state = self.initial_state(self.parameters)
        return self

    @staticmethod
    def initial_state(parameters: Parameters = None) -> SimState:
        sentence = "HELLO_WORLD"  # TODO abstract out with sim_parameters.json
        target_symbol = sentence[0]  # TODO use parameters.get('spelled_letters_count')
        default_criterion: List[SimDecisionCriteria] = [MaxIterationsSim(50), ProbThresholdSim(0.8)]
        init_stimuli = random.sample(alphabet(), 10)

        return SimState(target_symbol=target_symbol, current_sentence="", target_sentence=sentence,
                        display_alphabet=init_stimuli, inquiry_n=0, series_n=0,
                        series_results=[[]], decision_criterion=default_criterion)


def format_sim_state_dump(state: SimState):
    """ Formats a SimState dump to look similar to session.json """

    # TODO finish function

    ret = state.to_json()

    # removing the 'series_results' field and replacing with formatted 'series' dict
    ret.pop('series_results')
    series_dict = {}
    for i, series in enumerate(state.series_results):
        curr_series_dict = {}
        series_dict[str(i + 1)] = curr_series_dict
        for inq_idx, inq in enumerate(series):
            inq_dict = inq.to_json()
            curr_series_dict[str(inq_idx)] = inq_dict
    ret['series'] = series_dict

    return ret
