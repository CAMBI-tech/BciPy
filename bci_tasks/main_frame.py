from helpers.stim_gen import random_rsvp_calibration_seq_gen, \
    n_best_case_rsvp_seq_gen
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


class EvidenceFusion(object):
    """ Fuses likelihood evidences provided by the inference
        Attr:
            evidence_history(dict{list[ndarray]}): Dictionary of difference
                evidence types in list. Lists are ordered using the arrival
                time.
            likelihood(ndarray[]): current probability distribution over the
                set. Gets updated once new evindence arrives. """

    def __init__(self, list_name_evidence, len_dist):
        self.evidence_history = {name: [] for name in list_name_evidence}
        self.likelihood = np.ones(len_dist) / len_dist

    def update_and_fuse(self, dict_evidence):
        """ Updates the probability distribution
            Args:
                dict_evidence(dict[name: ndarray[float]]): dictionary of
                    evidences (EEG and other likelihoods)
                """
        names = dict_evidence.keys()
        values = dict_evidence.values()

        for name, value in zip(names, values):
            (self.evidence_history[name]).append(value)

        # TODO: Current rule is to multiply
        for value in values:
            self.likelihood *= value

        self.likelihood = self.likelihood / np.sum(self.likelihood)
        return self.likelihood

    def reset_history(self):
        # Makes all the values
        for value in self.evidence_history.values():
            del value[:]
        self.likelihood = np.ones(len(self.likelihood)) / len(self.likelihood)


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
            list_epoch(list[epoch]): List of stimuli in each sequence
                epoch(dict{items}):
                    - target(str): target of the epoch
                    - time_spent(ndarray[float]): |num_trials|x1
                      time spent on the sequence
                    - list_sti(list[list[str]]): presented symbols in each
                      sequence
                    - list_distribution(list[ndarray[float]]): list of |alp|x1
                        arrays with prob. dist. over alp
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

    def __init__(self, state=''):
        self.state = state
        self.displayed_state = form_display_state(state)

        # TODO: read from parameters file
        self.alphabet = list(string.ascii_uppercase) + ['<'] + ['_']

        self.list_epoch = [{'target': None, 'time_spent': 0, 'list_sti':
            [], 'list_distribution': []}]
        self.time = 0

    def decide(self, p):
        """ Once evidence is collected, decision_maker makes a decision to
            stop or not by leveraging the information of the stopping
            criteria. Can decide to do an epoch or schedule another sequence.
            Args:
                p(ndarray[float]): |A| x 1 distribution array
            Return:
                commitment(bin): True if a letter is a commitment is made
                                 False if requires more evidence
                args(dict[]): Extra arguments depending on the decision
                """

        self.list_epoch[-1]['list_distribution'].append(p)

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
                     not (np.max(p) > posterior_commit_threshold)):
            stimuli = self.schedule_sequence()
            commitment = False
            return commitment, {'stimuli': stimuli}
        else:
            self.do_epoch()
            commitment = True
            return commitment

    def do_epoch(self):
        """ Epoch refers to a commitment to a decision.
            If made, state is updated, displayed state is updated
            a new epoch is appended. """
        decision = self.decide_state_update()
        self.state = self.state + decision
        self.displayed_state = form_display_state(self.state)

        # Initialize next epoch
        self.list_epoch.append({'target': None, 'time_spent': 0, 'list_sti':
            [], 'list_distribution': []})

    def schedule_sequence(self):
        """ Schedules next sequence """
        stimuli = self.prepare_stimuli()
        self.list_epoch[-1]['list_sti'].append(stimuli)

        return stimuli

    def decide_state_update(self):
        """ Checks stopping criteria to commit to an epoch """
        idx = np.where(self.list_epoch[-1]['[op'] ==
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
        stimuli = \
            n_best_case_rsvp_seq_gen(self.alphabet, self.posteriors[-1][-1],
                                     num_sti=1)
        return stimuli

    def save_sequence_info(self):
        return 0


def _demo_fusion():
    len_alp = 4
    evidence_names = ['LM', 'ERP', 'FRP']
    num_epochs = 4
    num_sequences = 10

    conjugator = EvidenceFusion(evidence_names, len_dist=len_alp)

    print('Random Epochs!')
    for idx_ep in range(num_epochs):
        prior = np.abs(np.random.randn(len_alp))
        prior = prior / np.sum(prior)
        conjugator.update_and_fuse({'LM': prior})
        for idx in range(num_sequences):
            # Generate random sequences
            evidence_erp = 10 * np.abs(np.random.randn(len_alp))
            evidence_frp = 10 * np.abs(np.random.randn(len_alp))
            conjugator.update_and_fuse(
                {'ERP': evidence_erp, 'FRP': evidence_frp})
        print('Epoch: {}'.format(idx_ep))
        print(conjugator.evidence_history['ERP'])
        print(conjugator.evidence_history['FRP'])
        print(conjugator.evidence_history['LM'])
        print('Posterior:{}'.format(conjugator.likelihood))

        # Reset the conjugator before starting a new epoch for clear history
        conjugator.reset_history()


def main():
    _demo_fusion()

    return 0


if __name__ == "__main__":
    main()
