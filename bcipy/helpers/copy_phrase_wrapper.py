"""Defines the CopyPhraseWrapper."""
from typing import List, Tuple

import logging
import numpy as np
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.exceptions import BciPyCoreException
from bcipy.helpers.language_model import (
    histogram,
    with_min_prob,
)
from bcipy.helpers.stimuli import InquirySchedule, StimuliOrder, TrialReshaper
from bcipy.helpers.task import BACKSPACE_CHAR
from bcipy.signal.model import SignalModel
from bcipy.signal.process import get_default_transform
from bcipy.task.control.handler import DecisionMaker, EvidenceFusion
from bcipy.task.control.query import NBestStimuliAgent
from bcipy.task.control.criteria import (
    CriteriaEvaluator,
    MaxIterationsCriteria,
    MinIterationsCriteria,
    ProbThresholdCriteria,
)
from bcipy.task.data import EvidenceType
from bcipy.language.main import LanguageModel


log = logging.getLogger(__name__)


class CopyPhraseWrapper:
    """Basic copy phrase task duty cycle wrapper. Coordinates activities around
    spelling tasks, including:

    - Evidence management: adding and fusing evidence using the EvidenceFusion
    module.
    - Decision-making: uses evidence to make a decision or decide to continue
    by providing another inquiry (DecisionMaker).
    - Determining when to stop an inquiry and make a decision (StoppageCriteria).
    - Generation of inquiries.
    - Coordination with the Language Model.
    - Preparing EEG data for the SignalModel for classification.

    Parameters
    ----------
    - min_num_inq: The minimum number of inquiries to be displayed
    - max_num_inq: The maximum number of inquiries to be displayed
    - signal_model: model trained using a calibration session of the same user.
    - fs: sampling frequency
    - k: down sampling rate
    - alp: symbol set of the task
    - evidence_names: list of evidence types used for decision-making
    - task_list: list[(phrases, initial_states)] for the copy phrase task
    - lmodel: language model used (when 'LM') evidence type is used.
    - is_txt_stim: Whether or not the stimuli are text objects
    - device_name: name of the EEG device
    - device_channels: list of device channel names
    - decision_threshold: Minimum likelihood value required for a decision
    - backspace_prob: default language model probability for the
    backspace character.
    - backspace_always_shown: whether or not the backspace should
    always be presented.
    - filter_high: filter setting used when evaluating EEG data
    - filter_low: filter setting used when evaluating EEG data
    - filter_order: filter setting used when evaluating EEG data
    - notch_filter_frequency: filter setting used when evaluating EEG data
    - stim_length(int): the number of stimuli to present in each inquiry
    - stim_timing: seconds each stimuli is displayed; used for inquiry
    generation
    - stim_jitter: seconds that inquiry stimuli should be jittered (-/+ stim_timing[-1])
    """

    def __init__(self,
                 min_num_inq: int,
                 max_num_inq: int,
                 lmodel: LanguageModel,
                 signal_model: SignalModel = None,
                 fs: int = 300,
                 k: int = 2,
                 alp: List[str] = None,
                 evidence_names: List[EvidenceType] = [
                     EvidenceType.LM, EvidenceType.ERP
                 ],
                 task_list: List[Tuple[str, str]] = [('I_LOVE_COOKIES',
                                                      'I_LOVE_')],
                 is_txt_stim: bool = True,
                 device_name: str = 'LSL',
                 device_channels: List[str] = None,
                 decision_threshold: float = 0.8,
                 backspace_prob: float = 0.05,
                 backspace_always_shown: bool = False,
                 filter_high: int = 45,
                 filter_low: int = 2,
                 filter_order: int = 2,
                 notch_filter_frequency: int = 60,
                 stim_timing: List[float] = [1, .2],
                 stim_length: int = 10,
                 stim_jitter: float = 0,
                 stim_order: StimuliOrder = StimuliOrder.RANDOM):

        self.lmodel = lmodel
        self.conjugator = EvidenceFusion(evidence_names, len_dist=len(alp))

        inq_constants = []
        if backspace_always_shown and BACKSPACE_CHAR in alp:
            inq_constants.append(BACKSPACE_CHAR)

        # Stimuli Selection Module
        stopping_criteria = CriteriaEvaluator(
            continue_criteria=[MinIterationsCriteria(min_num_inq)],
            commit_criteria=[MaxIterationsCriteria(max_num_inq),
                             ProbThresholdCriteria(decision_threshold)])

        self.stim_length = stim_length
        self.stim_order = stim_order
        stimuli_agent = NBestStimuliAgent(alphabet=alp,
                                          len_query=self.stim_length)

        self.decision_maker = DecisionMaker(
            stimuli_agent=stimuli_agent,
            stopping_evaluator=stopping_criteria,
            state=task_list[0][1],
            alphabet=alp,
            is_txt_stim=is_txt_stim,
            stimuli_jitter=stim_jitter,
            stimuli_timing=stim_timing,
            stimuli_order=self.stim_order,
            inq_constants=inq_constants)

        self.alp = alp
        # non-letter target labels include the fixation cross and calibration.
        self.nonletters = ['+', 'PLUS', 'calibration_trigger']
        self.valid_targets = set(self.alp)

        self.signal_model = signal_model
        self.sampling_rate = fs
        self.downsample_rate = k
        self.filter_high = filter_high
        self.filter_low = filter_low
        self.filter_order = filter_order
        self.notch_filter_frequency = notch_filter_frequency

        self.mode = 'copy_phrase'
        self.task_list = task_list
        self.channel_map = analysis_channels(device_channels, device_name)
        self.backspace_prob = backspace_prob

    def evaluate_inquiry(
            self, raw_data: np.array, triggers: List[Tuple[str, float]],
            target_info: List[str], window_length: float
    ) -> Tuple[bool, Tuple[List[str], List[float], List[str]]]:
        """Once data is collected, infers meaning from the data and attempt to
        make a decision.

        Parameters
        ----------
        - raw_data: C x L eeg data where C is number of channels and L is the
        signal length
        - triggers: triggers e.g. `('A', 1)` as letter and flash time for the
        letter
        - target_info: target information about the stimuli
        - window_length: The length of the time between stimuli presentation

        Returns
        -------
        - (True, None) when commitment is made.
        - (False, next set of stimuli) when not enough evidence has
        been provided and stoppage criteria is not yet met.
        """
        lik_r = self.evaluate_eeg_evidence(raw_data, triggers, target_info,
                                           window_length)
        self.add_evidence(EvidenceType.ERP, lik_r)
        return self.decide()

    def evaluate_eeg_evidence(self, raw_data: np.array,
                              triggers: List[Tuple[str, float]],
                              target_info: List[str],
                              window_length: float) -> np.array:
        """Once data is collected, infers meaning from the data and return the results.

        Parameters
        ----------
        - raw_data: C x L eeg data where C is number of channels and L is the
        signal length
        - triggers: triggers e.g. `('A', 1)` as letter and flash time for the
        letter
        - target_info: target information about the stimuli
        - window_length: The length of the time between stimuli presentation

        Returns
        -------
        np.array of likelihood evidence
        """
        letters, times, target_info = self.letter_info(triggers, target_info)

        default_transform = get_default_transform(
            sample_rate_hz=self.sampling_rate,
            notch_freq_hz=self.notch_filter_frequency,
            bandpass_low=self.filter_low,
            bandpass_high=self.filter_high,
            bandpass_order=self.filter_order,
            downsample_factor=self.downsample_rate,
        )

        data, transformed_sample_rate = default_transform(raw_data, self.sampling_rate)

        # The data from DAQ is assumed to have offsets applied
        data, _ = TrialReshaper()(
            trial_targetness_label=target_info,
            timing_info=times,
            eeg_data=data,
            sample_rate=transformed_sample_rate,
            channel_map=self.channel_map,
            poststimulus_length=window_length)

        return self.signal_model.predict(data, letters, self.alp)

    def add_evidence(self, evidence_type: EvidenceType,
                     evidence: List[float]) -> np.array:
        """Add evidence to the conjugator.

        Parameters
        ----------
        - evidence_type : type of evidence (ex. `'LM'`, `'ERP'`)
        - evidence : ndarray[float], evidence for each stim

        Returns
        -------
        updated likelihoods after fusing the new evidence
        """
        assert evidence_type in self.conjugator.evidence_history.keys(
        ), f"Copy Phrase wrapper was not initialized with evidence type: {evidence_type}."
        return self.conjugator.update_and_fuse(
            {evidence_type: np.array(evidence)})

    def decide(self) -> Tuple[bool, InquirySchedule]:
        """Make a decision based on the current evidence.

        Returns
        -------
        - (True, None) when commitment is made.
        - (False, next set of stimuli) when not enough evidence has
        been provided and stoppage criteria is not yet met.
        """
        decision, new_stim = self.decision_maker.decide(
            self.conjugator.likelihood[:])
        return decision, new_stim

    def letter_info(self, triggers: List[Tuple[str, float]],
                    target_info: List[str]
                    ) -> Tuple[List[str], List[float], List[str]]:
        """
        Filters out non-letters and separates timings from letters.

        Parameters
        ----------
        - triggers: triggers e.g. [['A', 0.5], ...]
        as letter and flash time for the letter
        - target_info: target information about the stimuli;
        ex. ['nontarget', 'nontarget', ...]

        Returns
        -------
        (letters, times, target_info)
        """
        letters = []
        times = []
        target_types = []

        for i, (letter, stamp) in enumerate(triggers):
            if letter not in self.nonletters:
                letters.append(letter)
                times.append(stamp)
                target_types.append(target_info[i])

        # Raise an error if the stimuli includes unexpected terms
        if not set(letters).issubset(self.valid_targets):
            invalid = set(letters).difference(self.valid_targets)
            error_message = f'unexpected letters received in copy phrase: {invalid}'
            log.error(error_message)
            raise BciPyCoreException(error_message)

        return letters, times, target_types

    def initialize_series(self) -> Tuple[bool, InquirySchedule]:
        """If a decision is made initializes the next series."""
        assert self.lmodel, "Language model must be initialized."

        try:
            # First, reset the history for this new series
            self.conjugator.reset_history()

            # Get the displayed state
            update = self.decision_maker.displayed_state
            log.info(f"Querying language model: '{update}'")

            # update the lmodel and get back the priors
            lm_letter_prior = self.lmodel.predict(list(update))

            if BACKSPACE_CHAR in self.alp:
                # Apply configured backspace probability.
                sym = (BACKSPACE_CHAR, self.backspace_prob)
                lm_letter_prior = with_min_prob(lm_letter_prior, sym)

            # convert to format needed for evidence fusion;
            # probability value only in alphabet order.
            prior = [
                prior_prob for alp_letter in self.alp
                for prior_sym, prior_prob in lm_letter_prior
                if alp_letter == prior_sym
            ]

            # display histogram of LM probabilities
            log.debug(histogram(lm_letter_prior))

            # Try fusing the lmodel evidence
            try:
                prob_dist = self.conjugator.update_and_fuse(
                    {EvidenceType.LM: np.array(prior)})
            except Exception as fusion_error:
                log.exception(f'Error fusing language model evidence!: {fusion_error}')
                raise BciPyCoreException(fusion_error) from fusion_error

            # Get decision maker to give us back some decisions and stimuli
            is_accepted, sti = self.decision_maker.decide(prob_dist)

        except Exception as init_series_error:
            log.exception(f'Error in initialize_series: {init_series_error}')
            raise BciPyCoreException(init_series_error) from init_series_error

        return is_accepted, sti
