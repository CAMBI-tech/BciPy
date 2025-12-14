"""Task control handler module for BCI tasks.

This module provides classes for managing decision making and evidence fusion
in BCI tasks. It includes functionality for scheduling inquiries, managing
task state, and making decisions based on accumulated evidence.
"""

import logging
import string
from typing import Dict, List, Optional, Tuple

import numpy as np

from bcipy.config import SESSION_LOG_FILENAME
from bcipy.core.stimuli import InquirySchedule, StimuliOrder, inq_generator
from bcipy.core.symbols import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.task.control.criteria import CriteriaEvaluator
from bcipy.task.control.query import RandomStimuliAgent, StimuliAgent
from bcipy.task.data import EvidenceType

log = logging.getLogger(SESSION_LOG_FILENAME)


class EvidenceFusion:
    """Class for fusing likelihood evidence from multiple sources.

    This class manages the combination of evidence from different sources
    (e.g., EEG, eye tracking) to compute a final probability distribution
    over possible decisions.

    Attributes:
        evidence_history: Dictionary mapping evidence types to their history.
        likelihood: Current probability distribution over the decision space.
    """

    def __init__(self, list_name_evidence: List[EvidenceType],
                 len_dist: int) -> None:
        """Initialize the evidence fusion system.

        Args:
            list_name_evidence: List of evidence types to track.
            len_dist: Length of the probability distribution (number of
                possible decisions).
        """
        self.evidence_history: Dict[EvidenceType, List[np.ndarray]] = {
            name: [] for name in list_name_evidence
        }
        self.likelihood = np.ones(len_dist) / len_dist

    def update_and_fuse(self,
                        dict_evidence: Dict[EvidenceType,
                                            np.ndarray]) -> np.ndarray:
        """Update and fuse probability distributions with new evidence.

        Args:
            dict_evidence: Dictionary mapping evidence types to their
                likelihood arrays.

        Returns:
            np.ndarray: Updated probability distribution after fusion.
        """
        for key in dict_evidence:
            tmp = dict_evidence[key][:][:]
            self.evidence_history[key].append(tmp)

        # Current fusion rule is multiplication
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

    def reset_history(self) -> None:
        """Clears evidence history."""
        for value in self.evidence_history.values():
            del value[:]
        self.likelihood = np.ones(len(self.likelihood)) / len(self.likelihood)

    def save_history(self) -> None:
        """Save the current likelihood history.

        Note:
            Not currently implemented.
        """
        log.warning('save_history not implemented')

    @property
    def latest_evidence(self) -> Dict[EvidenceType, List[float]]:
        """Get the latest evidence of each type.

        Returns:
            Dict mapping evidence types to their most recent values.
        """
        return {
            name: list(evidence[-1]) if evidence else []
            for name, evidence in self.evidence_history.items()
        }


class DecisionMaker:
    """Scheduler and decision maker for BCI task control.

    This class manages the scheduling of inquiries and decision making based
    on accumulated evidence. It maintains the task state and coordinates
    the interaction between evidence collection and decision making.

    Attributes:
        state: Current state string, growing by 1 after each inquiry.
        displayed_state: State formatted for display.
        alphabet: List of possible symbols.
        is_txt_stim: Whether stimuli are text or images.
        stimuli_timing: Timing parameters for stimuli presentation.
        stimuli_order: Order of stimuli presentation.
        stimuli_jitter: Jitter in stimulus timing.
        inq_constants: Symbols to include in every inquiry.
        stopping_evaluator: Evaluator for stopping criteria.
        stimuli_agent: Agent for selecting stimuli.
        list_series: List of series data.
        time: Current time.
        inquiry_counter: Number of inquiries made.
        last_selection: Last selected symbol.
    """

    def __init__(
            self,
            state: str = '',
            alphabet: List[str] = list(string.ascii_uppercase) + [BACKSPACE_CHAR] +
            [SPACE_CHAR],
            is_txt_stim: bool = True,
            stimuli_timing: List[float] = [1, .2],
            stimuli_jitter: float = 0,
            stimuli_order: StimuliOrder = StimuliOrder.RANDOM,
            inq_constants: Optional[List[str]] = None,
            stopping_evaluator: Optional[CriteriaEvaluator] = None,
            stimuli_agent: Optional[StimuliAgent] = None) -> None:
        """Initialize the decision maker.

        Args:
            state: Initial state string.
            alphabet: List of possible symbols.
            is_txt_stim: Whether stimuli are text or images.
            stimuli_timing: [fixation_time, stimuli_flash_time].
            stimuli_jitter: Jitter in stimulus timing (seconds).
            stimuli_order: Order of stimuli presentation.
            inq_constants: Symbols to include in every inquiry.
            stopping_evaluator: Evaluator for stopping criteria.
            stimuli_agent: Agent for selecting stimuli.
        """
        self.state = state
        self.displayed_state = self.form_display_state(state)
        self.stimuli_timing = stimuli_timing
        self.stimuli_order = stimuli_order
        self.stimuli_jitter = stimuli_jitter

        self.alphabet = alphabet
        self.is_txt_stim = is_txt_stim

        self.list_series = [{
            'target': None,
            'time_spent': 0,
            'list_sti': [],
            'list_distribution': [],
            'decision': None
        }]
        self.time = 0
        self.inquiry_counter = 0

        self.stopping_evaluator = stopping_evaluator
        self.stimuli_agent = stimuli_agent or RandomStimuliAgent(
            alphabet=self.alphabet)
        self.last_selection = ''

        # Items shown in every inquiry
        self.inq_constants = inq_constants

    def reset(self, state: str = '') -> None:
        """Reset the decision maker to initial state.

        Args:
            state: New initial state string.
        """
        self.state = state
        self.displayed_state = self.form_display_state(self.state)

        self.list_series = [{
            'target': None,
            'time_spent': 0,
            'list_sti': [],
            'list_distribution': []
        }]
        self.time = 0
        self.inquiry_counter = 0

        self.stimuli_agent.reset()

    def form_display_state(self, state: str) -> str:
        """Format state string for display.

        Processes special characters (backspace, dots) and formats the
        state appropriately for display.

        Args:
            state: Raw state string.

        Returns:
            str: Formatted state string for display.
        """
        tmp = ''
        for i in state:
            if i == BACKSPACE_CHAR:
                tmp = tmp[0:-1]
            elif i != '.':
                tmp += i
        return tmp

    def update(self, state: str = '') -> None:
        """Update the current state.

        Args:
            state: New state string.
        """
        self.state = state
        self.displayed_state = self.form_display_state(state)

    def decide(self, p: np.ndarray) -> Tuple[bool, Optional[InquirySchedule]]:
        """Make a decision based on current evidence.

        Evaluates whether to commit to a decision or schedule another
        inquiry based on the current probability distribution and
        stopping criteria.

        Args:
            p: Probability distribution over possible decisions.

        Returns:
            Tuple containing:
                - bool: True if committing to a decision.
                - Optional[InquirySchedule]: Schedule for next inquiry if needed.
        """
        self.list_series[-1]['list_distribution'].append(p[:])

        if self.stopping_evaluator.should_commit(self.list_series[-1]):
            self.do_series()
            return True, None
        else:
            stimuli = self.schedule_inquiry()
            return False, stimuli

    def do_series(self) -> None:
        """Handle commitment to a decision.

        Updates state and prepares for the next series when a decision
        is made.
        """
        self.inquiry_counter = 0
        decision = self.decide_state_update()
        self.last_selection = decision
        self.state += decision
        self.displayed_state = self.form_display_state(self.state)

        self.list_series.append({
            'target': None,
            'time_spent': 0,
            'list_sti': [],
            'list_distribution': []
        })

        self.stimuli_agent.do_series()
        self.stopping_evaluator.do_series()

    def schedule_inquiry(self) -> InquirySchedule:
        """Schedule the next inquiry.

        Returns:
            InquirySchedule: Schedule for the next inquiry.
        """
        self.state += '.'
        stimuli = self.prepare_stimuli()
        self.list_series[-1]['list_sti'].append(stimuli[0])
        self.inquiry_counter += 1
        return stimuli

    def decide_state_update(self) -> str:
        """Determine the next state update.

        Returns:
            str: Selected symbol for state update.
        """
        idx = np.where(
            self.list_series[-1]['list_distribution'][-1] ==
            np.max(self.list_series[-1]['list_distribution'][-1]))[0][0]
        decision = self.alphabet[idx]
        self.list_series[-1]['decision'] = decision
        return decision

    def prepare_stimuli(self) -> InquirySchedule:
        """Prepare stimuli for the next inquiry.

        Returns:
            InquirySchedule: Schedule containing stimuli and timing information.
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
