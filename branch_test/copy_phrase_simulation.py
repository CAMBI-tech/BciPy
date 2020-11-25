""" This script simulates a copy phrase task. """
import string
import numpy as np
import scipy.io as sio
from scipy.stats import iqr
from sklearn.neighbors.kde import KernelDensity
from tqdm import tqdm
from copy import copy

from bcipy.tasks.rsvp.main_frame import DecisionMaker, EvidenceFusion
from bcipy.helpers.task import SPACE_CHAR, BACKSPACE_CHAR
from bcipy.tasks.rsvp.query_mechanisms import RandomStimuliAgent, NBestStimuliAgent, \
    MomentumStimuliAgent

from branch_test.synthesized_user import BinaryRSVPSynth

eps = np.power(.1, 6)

# Path to your local file (some prerecorded and pre-processed EEG data)
# Prospective file should follow the older Matlab version format
path = "./"
filename = "sample_user_1.mat"

delta = 5  # Manually shift the samples to increase AUC (caveman solution)
max_num_iter = 1000  # number of typing iterations with the system
lm_flag = False  # language model is active if True
len_query = 8  # number of trials in a sequence

# define the alphabet
alphabet = list(string.ascii_uppercase) + [BACKSPACE_CHAR] + [SPACE_CHAR]
len_alphabet = len(alphabet)

# evidence names for the system
evidence_names = ['LM', 'ERP']

# initial and target phrases for the typing simulation
list_phrase = ['I_LO', 'IT_O']
target_phrase = ['I_LOVE_COOKIES', 'IT_OCCURED']

if lm_flag:
    # TODO: load the language model here as l_model
    raise NotImplementedError("Language model option is not implemented!")

conjugator = EvidenceFusion(evidence_names, len_dist=len_alphabet)

# load data (x) and labels (y)
tmp = sio.loadmat(path + filename)
x = tmp['scores']
y = tmp['trialTargetness']

# remove data outliers with a threshold (caveman solution)
sc_threshold = 10000
y = y[x > -sc_threshold]
x = x[x > -sc_threshold]
y = y[x < sc_threshold]
x = x[x < sc_threshold]

# shift the positive class
x[y == 1] += delta

for idx_phrase in range(len(list_phrase)):

    # Initialize the synthetic user
    pre_phrase = list_phrase[idx_phrase]
    phrase = target_phrase[idx_phrase]
    synth = BinaryRSVPSynth(x=x, y=y, phrase=phrase, alp=alphabet,
                            erase_command=BACKSPACE_CHAR)

    # Initialize the decision maker
    decision_maker = DecisionMaker(state='', alphabet=alphabet,
                                   min_num_seq=2, max_num_seq=10,
                                   query_agent=MomentumStimuliAgent(
                                       alphabet=alphabet,
                                       len_query=len_query))

    # this is the dummy eeg_modelling part
    # currently it assumes the generative model and user EEG match perfectly
    bandwidth = 1.06 * min(np.std(x),
                           iqr(x) / 1.34) * np.power(x.shape[0], -0.2)
    classes = np.unique(y)
    cls_dep_x = [x[np.where(y == classes[i])] for i in range(len(classes))]

    dist_evidence = []
    for i in range(len(classes)):
        dist_evidence.append(KernelDensity(bandwidth=bandwidth))

        dat = np.expand_dims(cls_dep_x[i], axis=1)
        dist_evidence[i].fit(dat)

    for idx in tqdm(range(max_num_iter)):

        decision_maker.reset(state=pre_phrase)
        synth.reset()
        synth.update_state(decision_maker.displayed_state)

        seq_till_correct = [0] * len(phrase)
        d_counter = 0
        while decision_maker.displayed_state != phrase:

            # get prior information from language model
            tmp_displayed_state = "_" + decision_maker.displayed_state
            # use language model if language model flag is on
            if lm_flag:
                # If language model is present update the prior
                # Initialize the backspace probability
                idx_final_space = len(tmp_displayed_state) - \
                                  list(tmp_displayed_state)[
                                  ::-1].index("_") - 1
                l_model.set_reset(
                    tmp_displayed_state[idx_final_space + 1:])
                l_model.set_reset(
                    decision_maker.displayed_state.replace('_', ' '))
                lm_prior = l_model.get_prob()
            else:
                lm_prior = np.ones(len_alphabet) / len_alphabet

            prob = conjugator.update_and_fuse({evidence_names[0]: lm_prior})
            prob_ = np.array([i for i in prob])

            # Update the decision maker
            d, sti = decision_maker.decide(prob_)
            # Remove the fixation cross
            stimuli_letters = sti[0][0][1:]

            while True:

                # TODO: check if there are stimulus letters present
                # get answers from the user
                score = synth.answer(stimuli_letters)

                # get the likelihoods for the scores
                log_likelihood = []
                for i in score:
                    # This is where the dummy EEG modelling is used.
                    # You received responses (scores) from the synth user
                    # now get the likelihoods to run the Bayesian framework
                    dat = np.squeeze(i)
                    dens_0 = dist_evidence[0].score_samples(
                        dat.reshape(1, -1))[0]
                    dens_1 = dist_evidence[1].score_samples(
                        dat.reshape(1, -1))[0]
                    log_likelihood.append(np.asarray([dens_0, dens_1]))
                log_likelihood = np.array(log_likelihood)
                # compute likelihood ratio for the query
                lr = np.exp(log_likelihood[:, 1] - log_likelihood[:, 0])

                # initialize evidence with all ones
                evidence = np.ones(len_alphabet)

                c = 0
                # update evidence of the queries that are asked
                for q in stimuli_letters:
                    idx = alphabet.index(q)
                    evidence[idx] = lr[c]
                    c += 1

                # update posterior and decide what to do
                prob = conjugator.update_and_fuse({evidence_names[1]: evidence})
                prob_ = copy(prob)
                d, sti = decision_maker.decide(prob_)

                # TODO: add some visualizations

                if d:
                    # If a decision is made reset the evidence fusion and
                    # update the oracle using the system output
                    conjugator.reset_history()
                    synth.update_state(decision_maker.displayed_state)
                    break

                else:
                    # If not a commitment is made update the stimuli letters
                    stimuli_letters = sti[0][0][1:]
