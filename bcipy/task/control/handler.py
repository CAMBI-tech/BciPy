import logging
import string
from typing import Dict, List, Tuple, Optional

import numpy as np

from bcipy.helpers.stimuli import InquirySchedule, inq_generator, StimuliOrder
from bcipy.helpers.task import SPACE_CHAR, BACKSPACE_CHAR
from bcipy.task.control.query import RandomStimuliAgent, StimuliAgent
from bcipy.task.control.criteria import CriteriaEvaluator

log = logging.getLogger(__name__)


class EvidenceFusion():
    """ Fuses likelihood evidences provided by the inference
        Attr:
            evidence_history(dict{list[ndarray]}): Dictionary of difference
                evidence types in list. Lists are ordered using the arrival
                time.
            likelihood(ndarray[]): current probability distribution over the
                set. Gets updated once new evidence arrives. """

    def __init__(self, list_name_evidence, len_dist):
        self.evidence_history = {name: [] for name in list_name_evidence}
        self.likelihood = np.ones(len_dist) / len_dist

    def update_and_fuse(self, dict_evidence):
        """ Updates the probability distribution
            Args:
                dict_evidence(dict{name: ndarray[float]}): dictionary of
                    evidences (EEG and other likelihoods)
        """

        for key in dict_evidence.keys():
            tmp = dict_evidence[key][:][:]
            self.evidence_history[key].append(tmp)

        # Current rule is to multiply
        for value in dict_evidence.values():
            self.likelihood *= value[:]

        if np.isinf(np.sum(self.likelihood)):
            tmp = np.zeros(len(self.likelihood))
            tmp[np.where(self.likelihood == np.inf)[0][0]] = 1
            self.likelihood = tmp

        if not np.isnan(np.sum(self.likelihood)):
            self.likelihood = self.likelihood / np.sum(self.likelihood)

        likelihood = self.likelihood[:]

        return likelihood

    def reset_history(self):
        """ Clears evidence history """
        for value in self.evidence_history.values():
            del value[:]
        self.likelihood = np.ones(len(self.likelihood)) / len(self.likelihood)

    def save_history(self) -> None:
        """ Saves the current likelihood history """
        log.warning('save_history not implemented')
        return

    @property
    def latest_evidence(self) -> Dict[str, List[float]]:
        """Latest evidence of each type in the evidence history.

        Returns
        -------
        a dictionary with an entry for all configured evidence types.
        """
        return {
            name: list(evidence[-1]) if evidence else []
            for name, evidence in self.evidence_history.items()
        }


class DecisionMaker:
    """ Scheduler of the entire framework
        Attr:
            state(str): state of the framework, which increases in size
                by 1 after each inquiry. Elements are alphabet, ".,_,<"
                where ".": null_inquiry(no decision made)
                      "_": space bar
                      "<": back space
            alphabet(list[str]): list of symbols used by the framework. Can
                be switched with location of images or one hot encoded images.
            is_txt_stim(bool): whether the stimuli are text or images
            inq_constants(list[str]): optional list of letters which should appear in
                every inquiry.
            stopping_evaluator: CriteriaEvaluator - optional parameter to
                provide alternative rules for committing to a decision.
            stimuli_agent(StimuliAgent): the query selection mechanism of the
                system
            stimuli_timing(list[float]): list of timings for the stimuli ([fixation_time, stimuli_flash_time])
            stimuli_order(StimuliOrder): ordering of the stimuli (random, distributed)
            stimuli_jitter(float): jitter of the inquiry stimuli in seconds

        Functions:
            decide():
                Checks the criteria for making and series, using all
                evidences and decides to do an series or to collect more
                evidence
            do_series():
                Once committed an series perform updates to condition the
                distribution on the previous letter.
            schedule_inquiry():
                schedule the next inquiry using the current information
            decide_state_update():
                If committed to an series update the state using a decision
                metric.
                (e.g. pick the letter with highest likelihood)
            prepare_stimuli():
                prepares the query set for the next inquiry
                (e.g pick n-highest likely letters and randomly shuffle)
        """

    def __init__(self,
                 state: str = '',
                 alphabet: List[str] = list(string.ascii_uppercase) + [BACKSPACE_CHAR] + [SPACE_CHAR],
                 is_txt_stim: bool = True,
                 stimuli_timing: List[float] = [1, .2],
                 stimuli_jitter: float = 0,
                 stimuli_order: StimuliOrder = StimuliOrder.RANDOM,
                 inq_constants: Optional[List[str]] = None,
                 stopping_evaluator: CriteriaEvaluator = CriteriaEvaluator.default(min_num_inq=2,
                                                                                   max_num_inq=10,
                                                                                   threshold=0.8),
                 stimuli_agent: Optional[StimuliAgent] = None):
        self.state = state
        self.displayed_state = self.form_display_state(state)
        self.stimuli_timing = stimuli_timing
        self.stimuli_order = stimuli_order
        self.stimuli_jitter = stimuli_jitter

        self.alphabet = alphabet
        self.is_txt_stim = is_txt_stim

        self.list_series = [{'target': None, 'time_spent': 0,
                             'list_sti': [], 'list_distribution': [],
                             'decision': None}]
        self.time = 0
        self.inquiry_counter = 0

        self.stopping_evaluator = stopping_evaluator
        self.stimuli_agent = stimuli_agent or RandomStimuliAgent(alphabet=self.alphabet)
        self.last_selection = ''

        # Items shown in every inquiry
        self.inq_constants = inq_constants

    def reset(self, state=''):
        """ Resets the decision maker with the initial state
            Args:
                state(str): current state of the system """
        self.state = state
        self.displayed_state = self.form_display_state(self.state)

        self.list_series = [{'target': None, 'time_spent': 0,
                             'list_sti': [], 'list_distribution': []}]
        self.time = 0
        self.inquiry_counter = 0

        self.stimuli_agent.reset()

    def form_display_state(self, state):
        """ Forms the state information or the user that fits to the
            display. Basically takes '.' and BACKSPACE_CHAR into consideration and rewrites
            the state
            Args:
                state(str): state string
            Return:
                displayed_state(str): state without '<,.' and removes
                    backspaced letters """
        tmp = ''
        for i in state:
            if i == BACKSPACE_CHAR:
                tmp = tmp[0:-1]
            elif i != '.':
                tmp += i

        return tmp

    def update(self, state=''):
        self.state = state
        self.displayed_state = self.form_display_state(state)

    def decide(self, p) -> Tuple[bool, InquirySchedule]:
        """ Once evidence is collected, decision_maker makes a decision to
            stop or not by leveraging the information of the stopping
            criteria. Can decide to do an series or schedule another inquiry.

        Args
        ----
        p(ndarray[float]): |A| x 1 distribution array
            |A|: cardinality of the alphabet

        Return
        ------
        - commitment: True if a letter is a commitment is made
        False if requires more evidence
        - inquiry schedule: Extra arguments depending on the decision
        """

        self.list_series[-1]['list_distribution'].append(p[:])

        # Check stopping criteria
        if self.stopping_evaluator.should_commit(self.list_series[-1]):
            self.do_series()
            return True, None
        else:
            stimuli = self.schedule_inquiry()
            return False, stimuli

    def do_series(self):
        """ series refers to a commitment to a decision.
            If made, state is updated, displayed state is updated
            a new series is appended. """
        self.inquiry_counter = 0
        decision = self.decide_state_update()
        self.last_selection = decision
        self.state += decision
        self.displayed_state = self.form_display_state(self.state)

        # Initialize next series
        self.list_series.append({'target': None, 'time_spent': 0,
                                 'list_sti': [], 'list_distribution': []})

        self.stimuli_agent.do_series()
        self.stopping_evaluator.do_series()

    def schedule_inquiry(self) -> InquirySchedule:
        """ Schedules next inquiry """
        self.state += '.'
        stimuli = self.prepare_stimuli()
        self.list_series[-1]['list_sti'].append(stimuli[0])
        self.inquiry_counter += 1

        return stimuli

    def decide_state_update(self):
        """ Checks stopping criteria to commit to an series """
        idx = np.where(
            self.list_series[-1]['list_distribution'][-1] ==
            np.max(self.list_series[-1]['list_distribution'][-1]))[0][0]
        decision = self.alphabet[idx]
        self.list_series[-1]['decision'] = decision
        return decision

    def prepare_stimuli(self) -> InquirySchedule:
        """ Given the alphabet, under a rule, prepares a stimuli for
            the next inquiry.

        Return
        ------
        stimuli(tuple[list[char],list[float],list[str]]): tuple of
        stimuli information. [0]: letter, [1]: timing, [2]: color
        """

        # querying agent decides on possible letters to be shown on the screen
        query_els = self.stimuli_agent.return_stimuli(
            self.list_series[-1]['list_distribution'],
            constants=self.inq_constants)
        # once querying is determined, append with timing and color info.
        stimuli = inq_generator(query=query_els,
                                inquiry_count=1,
                                is_txt=self.is_txt_stim,
                                timing=self.stimuli_timing,
                                stim_order=self.stimuli_order,
                                stim_jitter=self.stimuli_jitter)
        return stimuli
