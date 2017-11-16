from helpers.stim_gen import random_rsvp_sequence_generator
import numpy as np
import string


def form_display_state(state):
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


def fusion():
    """ Fuses likelihood evidences provided by the """
    return 0


class DecisionMaker(object):
    """ Scheduler of the entire framework
        Attr:
            state(str): state of the framework, which increases in size
                by 1 after each sequence. Elements are alphabet, ".,_,<"
                where ".": null_sequence(no decision made)
                      "_": space bar
                      "<": back space
            displayed_state(str): visualization of the state to the user
                only includes the alphabet and "_"
            alphabet(list[str]): list of symbols used by the framework. Can
                be switched with location of images or one hot encoded images.
            posteriors(list[list[ndarray]]): list of epochs, where epochs
                are list of sequences, informs the current probability
                distribution of the sequence.
            time(float): system time
            evidence(list[str]): list of evidences used in the framework
            list_priority_evidence(list[]): priority list for the vidences
            sequence_counter(dict[str(val=float)]): number of sequences
                passed for each particular evidence
            epoch_counter(int): number of epochs through the session
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
                metric. (e.g. pick the letter with highest likelihood)
        """

    def __init__(self, state=''):
        self.state = state
        self.displayed_state = form_display_state(state)

        self.alphabet = list(string.ascii_uppercase) + ['<'] + ['_']

        self.posteriors = []
        self.time = 0

        self.evidence = []
        self.list_priority_evidence = []
        self.sequence_counter = {'ERP': 0, 'FRP0': 0}
        self.epoch_counter = 0

    def decide(self):

        # Stopping Criteria
        # TODO: Read from parameters
        min_num_seq = 1
        max_num_seq = 2
        time_threshold = 1  # in seconds
        posterior_commit_threshold = .8

        # Check stopping criteria
        if (sum(self.sequence_counter.values()) < min_num_seq) or \
                (not (sum(self.sequence_counter.values()) > max_num_seq) and
                     not (self.time > time_threshold) and
                     not (np.max(self.posteriors[-1][-1]) >
                              posterior_commit_threshold)):
            self.schedule_sequence()
        else:
            self.do_epoch()

    def do_epoch(self):
        decision = self.decide_state_update()
        self.state = self.state + decision
        self.displayed_state = form_display_state(self.state)
        # Append a new list for the next epoch
        self.posteriors.append([])

    def schedule_sequence(self):

        return 0

    def decide_state_update(self):
        """ Checks stopping criteria to commit to an epoch """
        idx = np.where(self.posteriors[-1] ==
                       np.max(self.posteriors[-1]))[0][0]
        decision = self.alphabet[idx]
        return decision

    def prepare_stimuli(self):
        """ Given the alphabet, under a rule, prepares a stimuli for
            the next sequence
            Return:
                stimuli(tuple[list[char],list[float],list[str]]): tuple of
                    stimuli information. [0]: letter, [1]: timing, [2]: color
                """
        stimuli = random_rsvp_sequence_generator(self.alphabet, num_sti=1)[0]
        return stimuli

    def save_sequence_info(self):
        return 0
