# -*- coding: utf-8 -*-
from bcipy.tasks.rsvp.stopping_criteria import CriteriaEvaluator
from bcipy.tasks.rsvp.query_mechanisms import RandomStimuliAgent
from bcipy.helpers.stimuli import rsvp_seq_generator
from bcipy.helpers.task import SPACE_CHAR
import logging
import numpy as np
import string

log = logging.getLogger(__name__)


class EvidenceFusion(object):
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

        # I
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

    def save_history(self):
        """ Saves the current likelihood history """
        return 0


class DecisionMaker:
    """ Scheduler of the entire framework
        Attr:
            decision_threshold: Minimum combined likelihood required for a
                decision
            state(str): state of the framework, which increases in size
                by 1 after each sequence. Elements are alphabet, ".,_,<"
                where ".": null_sequence(no decision made)
                      "_": space bar
                      "<": back space
            displayed_state(str): visualization of the state to the user
                only includes the alphabet and "_"
            alphabet(list[str]): list of symbols used by the framework. Can
                be switched with location of images or one hot encoded images.
            time(float): system time
            evidence(list[str]): list of evidences used in the framework
            list_priority_evidence(list[]): priority list for the evidences
            sequence_counter(dict[str(val=float)]): number of sequences
                passed for each particular evidence
            list_epoch(list[epoch]): List of stimuli in each sequence
                epoch(dict{items}):
                    - target(str): target of the epoch
                    - time_spent(ndarray[float]): |num_trials|x1
                      time spent on the sequence
                    - list_sti(list[list[str]]): presented symbols in each
                      sequence
                    - list_distribution(list[ndarray[float]]): list of |alp|x1
                        arrays with prob. dist. over alp
            seq_constants(list[str]): list of letters which should appear in
                every sequence.
            stopping_evaluator: CriteriaEvaluator - optional parameter to
                provide alternative rules for committing to a decision.
            stimuli_agent(StimuliAgent): the query selection mechanism of the
                system

        Functions:
            decide():
                Checks the criteria for making and epoch, using all
                evidences and decides to do an epoch or to collect more
                evidence
            do_epoch():
                Once committed an epoch perform updates to condition the
                distribution on the previous letter.
            schedule_sequence():
                schedule the next sequence using the current information
            decide_state_update():
                If committed to an epoch update the state using a decision
                metric.
                (e.g. pick the letter with highest likelihood)
            prepare_stimuli():
                prepares the query set for the next sequence
                (e.g pick n-highest likely letters and randomly shuffle)
        """

    def __init__(self,
                 state='',
                 alphabet=list(string.ascii_uppercase) + ['<'] + [SPACE_CHAR],
                 is_txt_stim=True,
                 stimuli_timing=[1, .2],
                 seq_constants=None,
                 stopping_evaluator=CriteriaEvaluator.default(min_num_seq=2,
                                                              max_num_seq=10,
                                                              threshold=0.8),
                 stimuli_agent=RandomStimuliAgent(
                     alphabet=list(string.ascii_uppercase) +
                              ['<'] + [SPACE_CHAR])):
        self.state = state
        self.displayed_state = self.form_display_state(state)
        self.stimuli_timing = stimuli_timing

        # TODO: read from parameters file
        self.alphabet = alphabet
        self.is_txt_stim = is_txt_stim

        self.list_epoch = [{'target': None, 'time_spent': 0,
                            'list_sti': [], 'list_distribution': [],
                            'decision': None}]
        self.time = 0
        self.sequence_counter = 0

        # Stopping Criteria
        self.stopping_evaluator = stopping_evaluator

        # Stimuli Agent
        self.stimuli_agent = stimuli_agent

        self.last_selection = ''

        # Items shown in every sequence
        self.seq_constants = seq_constants

    def reset(self, state=''):
        """ Resets the decision maker with the initial state
            Args:
                state(str): current state of the system """
        self.state = state
        self.displayed_state = self.form_display_state(self.state)

        self.list_epoch = [{'target': None, 'time_spent': 0,
                            'list_sti': [], 'list_distribution': []}]
        self.time = 0
        self.sequence_counter = 0

        self.stimuli_agent.reset()

    def form_display_state(self, state):
        """ Forms the state information or the user that fits to the
            display. Basically takes '.' and '<' into consideration and rewrites
            the state
            Args:
                state(str): state string
            Return:
                displayed_state(str): state without '<,.' and removes
                    backspaced letters """
        tmp = ''
        for i in state:
            if i == '<':
                tmp = tmp[0:-1]
            elif i != '.':
                tmp += i

        return tmp

    def update(self, state=''):
        self.state = state
        self.displayed_state = self.form_display_state(state)

    def decide(self, p):
        """ Once evidence is collected, decision_maker makes a decision to
            stop or not by leveraging the information of the stopping
            criteria. Can decide to do an epoch or schedule another sequence.
            Args:
                p(ndarray[float]): |A| x 1 distribution array
                    |A|: cardinality of the alphabet
            Return:
                commitment(bin): True if a letter is a commitment is made
                                 False if requires more evidence
                args(dict[]): Extra arguments depending on the decision
                """

        self.list_epoch[-1]['list_distribution'].append(p[:])

        # Check stopping criteria
        if self.stopping_evaluator.should_commit(self.list_epoch[-1]):
            self.do_epoch()
            return True, None
        else:
            stimuli = self.schedule_sequence()
            return False, stimuli

    def do_epoch(self):
        """ Epoch refers to a commitment to a decision.
            If made, state is updated, displayed state is updated
            a new epoch is appended. """
        self.sequence_counter = 0
        decision = self.decide_state_update()
        self.last_selection = decision
        self.state += decision
        self.displayed_state = self.form_display_state(self.state)

        # Initialize next epoch
        self.list_epoch.append({'target': None, 'time_spent': 0,
                                'list_sti': [], 'list_distribution': []})

        self.stimuli_agent.do_epoch()
        self.stopping_evaluator.do_epoch()

    def schedule_sequence(self):
        """ Schedules next sequence """
        self.state += '.'
        stimuli = self.prepare_stimuli()
        self.list_epoch[-1]['list_sti'].append(stimuli[0])
        self.sequence_counter += 1

        return stimuli

    def decide_state_update(self):
        """ Checks stopping criteria to commit to an epoch """
        idx = np.where(
            self.list_epoch[-1]['list_distribution'][-1] ==
            np.max(self.list_epoch[-1]['list_distribution'][-1]))[0][0]
        decision = self.alphabet[idx]
        self.list_epoch[-1]['decision'] = decision
        return decision

    def prepare_stimuli(self):
        """ Given the alphabet, under a rule, prepares a stimuli for
            the next sequence
            Return:
                stimuli(tuple[list[char],list[float],list[str]]): tuple of
                    stimuli information. [0]: letter, [1]: timing, [2]: color
                """

        # querying agent decides on possible letters to be shown on the screen
        query_els = self.stimuli_agent.return_stimuli(
            self.list_epoch[-1]['list_distribution'])
        # once querying is determined, append with timing and color info
        stimuli = rsvp_seq_generator(query=query_els,
                                     stim_number=1,
                                     is_txt=self.is_txt_stim,
                                     timing=self.stimuli_timing,
                                     seq_constants=self.seq_constants)
        return stimuli
