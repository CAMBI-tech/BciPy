from helpers.stim_gen import n_best_case_rsvp_seq_gen
import numpy as np
import string


# TODO: Decide if this method should be inside decision maker class!
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
            self.evidence_history[key].append(dict_evidence[key])

        # TODO: Current rule is to multiply
        for value in dict_evidence.values():
            self.likelihood *= value
        self.likelihood = self.likelihood / np.sum(self.likelihood)

        return self.likelihood

    def reset_history(self):
        """ Clears evidence history """
        for value in self.evidence_history.values():
            del value[:]
        self.likelihood = np.ones(len(self.likelihood)) / len(self.likelihood)

    def save_history(self):
        """ Saves the current likelihood history """
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

    def __init__(self, state='',
                 alphabet=list(string.ascii_uppercase) + ['<'] + ['_']):
        self.state = state
        self.displayed_state = form_display_state(state)

        # TODO: read from parameters file
        self.alphabet = alphabet

        self.list_epoch = [{'target': None, 'time_spent': 0, 'list_sti':
            [], 'list_distribution': []}]
        self.time = 0
        self.sequence_counter = 0

        # Stopping Criteria
        # TODO: Read from parameters
        self.min_num_seq = 1
        self.max_num_seq = 10
        self.posterior_commit_threshold = .8

    def update(self, state=''):
        self.state = state
        self.displayed_state = form_display_state(state)

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

        self.list_epoch[-1]['list_distribution'].append(p)

        # Check stopping criteria
        if self.sequence_counter < self.min_num_seq or \
                not (self.sequence_counter > self.max_num_seq
                     or np.max(p) > self.posterior_commit_threshold):

            stimuli = self.schedule_sequence()
            commitment = False
            return commitment, {'stimuli': stimuli}
        else:
            self.do_epoch()
            commitment = True
            return commitment, []

    def do_epoch(self):
        """ Epoch refers to a commitment to a decision.
            If made, state is updated, displayed state is updated
            a new epoch is appended. """
        self.sequence_counter = 0
        decision = self.decide_state_update()
        self.state += decision
        self.displayed_state = form_display_state(self.state)

        # Initialize next epoch
        self.list_epoch.append({'target': None, 'time_spent': 0, 'list_sti':
            [], 'list_distribution': []})

    def schedule_sequence(self):
        """ Schedules next sequence """
        self.state += '.'
        stimuli = self.prepare_stimuli()
        self.list_epoch[-1]['list_sti'].append(stimuli[0])
        self.sequence_counter += 1

        return stimuli

    def decide_state_update(self):
        """ Checks stopping criteria to commit to an epoch """
        idx = np.where(self.list_epoch[-1]['list_distribution'][-1] ==
                       np.max(self.list_epoch[-1]['list_distribution'][-1]))[
            0][0]
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
            n_best_case_rsvp_seq_gen(self.alphabet, self.list_epoch[-1][
                'list_distribution'][-1], num_sti=1)
        return stimuli


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


def _demo_decision_maker():
    alp = ['T', 'H', 'I', 'S', 'I', 'S', 'L', 'A', 'M', 'E']
    len_alp = len(alp)
    evidence_names = ['LM', 'ERP', 'FRP']
    num_epochs = 10

    conjugator = EvidenceFusion(evidence_names, len_dist=len_alp)
    decision_maker = DecisionMaker(state='', alphabet=alp)

    for idx_epoch in range(num_epochs):

        while True:
            # Generate random sequences
            evidence_erp = np.abs(np.random.randn(len_alp))
            evidence_erp[idx_epoch] += 1
            evidence_frp = np.abs(np.random.randn(len_alp))
            evidence_frp[idx_epoch] += 3

            p = conjugator.update_and_fuse(
                {'ERP': evidence_erp, 'FRP': evidence_frp})

            d, arg = decision_maker.decide(p)
            if d:
                break
        # Reset the conjugator before starting a new epoch for clear history
        conjugator.reset_history()

    print('State:{}'.format(decision_maker.state))
    print('Displayed State: {}'.format(decision_maker.displayed_state))


if __name__ == "__main__":
    _demo_decision_maker()
