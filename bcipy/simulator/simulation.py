from bcipy.bci_tasks.simulator.oracle import BinaryRSVPOracle
import scipy.io as sio
from bcipy.bci_tasks.main_frame import DecisionMaker, EvidenceFusion
from scipy.stats import iqr
from sklearn.neighbors.kde import KernelDensity
import string
import numpy as np


def progress_bar(iteration, total, prefix='', suffix='', decimals=1,
                 length=100, fill='-'):
    """ Progress bar can be used in any finite iteration.
        Args:
            """

    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    fill_length = int(length * iteration // total)
    bar = fill * fill_length + ' ' * (length - fill_length)
    print('\r{} {} %{} {}'.format(prefix, bar, percent, suffix)),
    if iteration == total:
        print('')


alp = list(string.ascii_uppercase) + ['<'] + ['_']
len_alp = len(alp)
evidence_names = ['LM', 'Eps']

# phrase that is already typed before the task
pre_phrase = 'CA'
# target of the task
phrase = "CAE"
backspace_prob = 1. / 35.
max_num_mc = 1000
min_num_seq = 30
max_num_seq = 30

# query_methods = [MomentumQuerying(alp=alp, gam=.9), NBestQuery(alp=alp)]
conjugator = EvidenceFusion(evidence_names, len_dist=len_alp)
decision_maker = DecisionMaker(state='', alphabet=alp)
decision_maker.min_num_seq = min_num_seq
decision_maker.max_num_seq = max_num_seq

tmp = sio.loadmat("./2a13a-s3-88.mat")
x = tmp['scores']
y = tmp['trialTargetness']

# data collected can have outlier samples. these samples will cause a problem once a
# KDE is fit. Therefore we discard samples which have unrealistically high score.
sc_threshold = 1000000

# modify data for outliers
y = y[x > -sc_threshold]
x = x[x > -sc_threshold]
y = y[x < sc_threshold]
x = x[x < sc_threshold]

# artificially shift mean of the positive class and separate means of the KDEs
x[y == 1] -= 25

oracle = BinaryRSVPOracle(x, y, phrase=phrase, alp=alp)

# this is the dummy eeg_modelling part

# select KDE bandwidth
# ref: Silverman, B.W. (1986). Density Estimation for Statistics and Data Analysis.
# London: Chapman & Hall/CRC. p. 48. ISBN 0-412-24620-1
bandwidth = 1.06 * min(np.std(x),
                       iqr(x) / 1.34) * np.power(x.shape[0], -0.2)
classes = np.unique(y)
cls_dep_x = [x[np.where(y == classes[i])] for i in range(len(classes))]
dist = []
for i in range(len(classes)):
    dist.append(KernelDensity(bandwidth=bandwidth))

    dat = np.expand_dims(cls_dep_x[i], axis=1)
    dist[i].fit(dat)

# number of sequences spent for correct decision (adds incorrect decisions)
seq_holder = [[], []]

target_dist = [[], []]
progress_bar(0, max_num_mc, prefix='Progress:', suffix='Complete', length=50)
bar_counter = 0
method_count = 0

sum_seq_till_correct = np.zeros(len(phrase))

for idx in range(max_num_mc):
    progress_bar(bar_counter + 1, max_num_mc, prefix='Progress:', suffix='Complete', length=50)

    decision_maker.reset(state=pre_phrase)
    oracle.reset()
    oracle.update_state(decision_maker.displayed_state)

    seq_till_correct = [0] * len(phrase)
    d_counter = 0
    while decision_maker.displayed_state != phrase:

        # can be used for artificial language model
        # get prior information from language model
        lm_prior = np.ones(len(alp))

        prob = conjugator.update_and_fuse({'LM': lm_prior})
        prob_new = np.array([i for i in prob])
        d, sti = decision_maker.decide(prob_new)

        while True:
            # get answers from the user
            score = oracle.answer(sti['stimuli'][0][0][1::])

            # get the likelihoods for the scores
            likelihood = []
            for i in score:
                dat = np.squeeze(i)
                dens_0 = dist[0].score_samples(dat)[0]
                dens_1 = dist[1].score_samples(dat)[0]
                likelihood.append(np.asarray([dens_0, dens_1]))
            likelihood = np.array(likelihood)
            # compute likelihood ratio for the query
            lr = np.exp(likelihood[:, 1] - likelihood[:, 0])

            # initialize evidence with all ones
            evidence = np.ones(len_alp)
            c = 0
            # update evidence of the queries that are asked
            for q in sti['stimuli'][0][0][1::]:
                idx = alp.index(q)
                evidence[idx] = lr[c]
                c += 1

            # update posterior and decide what to do
            prob = conjugator.update_and_fuse({'Eps': evidence})
            prob_new = np.array([i for i in prob])
            d, sti = decision_maker.decide(prob_new)

            # after decision update the user about the current delta
            oracle.update_state(decision_maker.displayed_state)

            print('\r State:{}'.format(decision_maker.state)),

            if d:
                tmp_dist = list(np.array(
                    decision_maker.list_epoch[-2]['list_distribution'])[:,
                                alp.index(oracle.state)])
                target_dist[method_count].append(tmp_dist)

                break
        # Reset the conjugator before starting a new epoch for clear history
        conjugator.reset_history()
        seq_till_correct[d_counter] += len(decision_maker.list_epoch[-2]['list_sti'])
        if (decision_maker.list_epoch[-2]['decision'] == phrase[d_counter] and
                decision_maker.displayed_state == phrase[0:len(
                    decision_maker.displayed_state)]):
            d_counter += 1


    seq_holder[method_count].append(seq_till_correct)
    bar_counter += 1

