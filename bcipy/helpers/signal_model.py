"""Defines the CopyPhraseWrapper."""
from typing import List, Tuple

import numpy as np

from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.task import BACKSPACE_CHAR, trial_reshaper
from bcipy.signal.model.inference import inference
from bcipy.signal.process.filter import bandpass, notch, downsample
from bcipy.tasks.rsvp.main_frame import EvidenceFusion, DecisionMaker
from bcipy.helpers.language_model import norm_domain, sym_appended, \
    equally_probable


class CopyPhraseWrapper:
    """Basic copy phrase task duty cycle wrapper.

    Given the phrases once operate() is called performs the task.
    Attr:
        min_num_seq: The minimum number of sequences to be displayed
        max_num_seq: The maximum number of sequences to be displayed
        model(pipeline): model trained using a calibration session of the
            same user.
        fs(int): sampling frequency
        k(int): down sampling rate
        alp(list[str]): symbol set of the task
        task_list(list[tuple(str,str)]): list[(phrases, initial_states)] for
            the copy phrase task
        is_txt_stim: Whether or not the stimuli are text objects
        conjugator(EvidenceFusion): fuses evidences in the task
        decision_maker(DecisionMaker): mastermind of the task
        mode(str): mode of thet task (should be copy phrase)
        d(binary): decision flag
        sti(list(tuple)): stimuli for the display
        decision_threshold: Minimum likelihood value required for a decision
        backspace_prob(float): default language model probability for the
            backspace character.
        backspace_always_shown(bool): whether or not the backspace should
            always be presented.
    """

    def __init__(self, min_num_seq, max_num_seq, signal_model=None, fs=300, k=2,
                 alp=None, evidence_names=['LM', 'ERP'],
                 task_list=[('I_LOVE_COOKIES', 'I_LOVE_')], lmodel=None,
                 is_txt_stim=True, device_name='LSL', device_channels=None,
                 stimuli_timing=[1, .2],
                 decision_threshold=0.8,
                 backspace_prob=0.05,
                 backspace_always_shown=False,
                 filter_high=45,
                 filter_low=2,
                 filter_order=2,
                 notch_filter_frequency=60):

        self.conjugator = EvidenceFusion(evidence_names, len_dist=len(alp))

        seq_constants = []
        if backspace_always_shown and BACKSPACE_CHAR in alp:
            seq_constants.append(BACKSPACE_CHAR)
        self.decision_maker = DecisionMaker(
            min_num_seq,
            max_num_seq,
            decision_threshold=decision_threshold,
            state=task_list[0][1],
            alphabet=alp,
            is_txt_stim=is_txt_stim,
            stimuli_timing=stimuli_timing,
            seq_constants=seq_constants)
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
        self.lmodel = lmodel
        self.channel_map = analysis_channels(device_channels, device_name)
        self.backspace_prob = backspace_prob

    def evaluate_sequence(self, raw_data, triggers, target_info, window_length, artifact_rejection=False):
        """Once data is collected, infers meaning from the data.

        Args:
            raw_data(ndarray[float]): C x L eeg data where C is number of
                channels and L is the signal length
            triggers(list[tuple(str,float)]): triggers e.g. ('A', 1)
                as letter and flash time for the letter
            target_info(list[str]): target information about the stimuli
            window_length(int): The length of the time between stimuli presentation
            artifact_rejection(bool): boolean indicating whether or not 
                artifact rejection should be applied.
        """
        letters, times, target_info = self.letter_info(triggers, target_info)

        # Remove 60hz noise with a notch filter
        notch_filter_data = notch.notch_filter(
            raw_data, self.sampling_rate, frequency_to_remove=self.notch_filter_frequency)

        # bandpass filter from 2-45hz
        filtered_data = bandpass.butter_bandpass_filter(
            notch_filter_data,
            self.filter_low,
            self.filter_high,
            self.sampling_rate,
            order=self.filter_order)

        # downsample
        data = downsample.downsample(
            filtered_data, factor=self.downsample_rate)
        x, _, _, _ = trial_reshaper(target_info, times, data, fs=self.sampling_rate,
                                    k=self.downsample_rate, mode=self.mode,
                                    channel_map=self.channel_map,
                                    trial_length=window_length)

        lik_r = inference(x, letters, self.signal_model, self.alp)
        prob = self.conjugator.update_and_fuse({'ERP': lik_r})
        decision, sti = self.decision_maker.decide(prob)

        return decision, sti

    def letter_info(self, triggers: List[Tuple[str, float]],
                    target_info: List[str]
                    ) -> Tuple[List[str], List[float], List[str]]:
        """
        Filters out non-letters and separates timings from letters.
        Parameters:
        -----------
         triggers: triggers e.g. [['A', 0.5], ...]
                as letter and flash time for the letter
         target_info: target information about the stimuli;
            ex. ['nontarget', 'nontarget', ...]
        Returns:
        --------
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
            raise Exception(
                f'unexpected letters received in copy phrase: {invalid}')

        return letters, times, target_types

    def initialize_epoch(self):
        """If a decision is made initializes the next epoch."""

        try:
            # First, reset the history for this new epoch
            self.conjugator.reset_history()

            # If there is no language model specified, mock the LM prior
            # TODO: is the probability domain correct? ERP evidence is in
            # the log domain; LM by default returns negative log domain.
            if not self.lmodel:
                # mock probabilities to be equally likely for all letters.
                overrides = {BACKSPACE_CHAR: self.backspace_prob}
                prior = equally_probable(self.alp, overrides)

            # Else, let's query the lmodel for priors
            else:
                # Get the displayed state
                # TODO: for oclm this should be a list of (sym, prob)
                update = self.decision_maker.displayed_state

                # update the lmodel and get back the priors
                lm_prior = self.lmodel.state_update(update)

                # normalize to probability domain if needed
                if getattr(self.lmodel, 'normalized', False):
                    lm_letter_prior = lm_prior['letter']
                else:
                    lm_letter_prior = norm_domain(lm_prior['letter'])

                if BACKSPACE_CHAR in self.alp:
                    # Append backspace if missing.
                    sym = (BACKSPACE_CHAR, self.backspace_prob)
                    lm_letter_prior = sym_appended(lm_letter_prior, sym)

                # convert to format needed for evidence fusion;
                # probability value only in alphabet order.
                # TODO: ensure that probabilities still add to 1.0
                prior = [prior_prob
                         for alp_letter in self.alp
                         for prior_sym, prior_prob in lm_letter_prior
                         if alp_letter == prior_sym]

            # Try fusing the lmodel evidence
            try:
                prob_dist = self.conjugator.update_and_fuse(
                    {'LM': np.array(prior)})
            except Exception as lm_exception:
                print("Error updating language model!")
                raise lm_exception

            # Get decision maker to give us back some decisions and stimuli
            is_accepted, sti = self.decision_maker.decide(prob_dist)


        except Exception as init_exception:
            print("Error in initialize_epoch: %s" % (init_exception))
            raise init_exception

        return is_accepted, sti
