import os
from branch_test.oracle import BinaryRSVPOracle
from branch_test.progress_bar import progress_bar
import scipy.io as sio
from bcipy.tasks.main_frame import DecisionMaker, EvidenceFusion
from scipy.stats import norm, iqr
from sklearn.neighbors.kde import KernelDensity
from bcipy.tasks.query_selection import RandomQuery, MomentumQuerying, \
    NBestQuery
import numpy as np
import string

# epsilon (a small number)
eps = np.power(.1, 7)

# System path parameters
# path to data (data should be in MATLAB)
path = "AL12311987_IRB130107_ERPCalibration.mat"

# maximum number of sequences and minimum number of sequences for decision
min_num_seq = 1
max_num_seq = 15
# number of trials in a sequence (bathced querying)
len_query = 5

alp = list(string.ascii_uppercase) + ['-'] + ['<']

# List of query methods. If required, can be populated
query_methods = [
    RandomQuery(len_alp=len(alp), len_query=len_query),
    NBestQuery(len_alp=len(alp), len_query=len_query),
    MomentumQuerying(len_alp=len(alp), len_query=len_query, gam=1, lam=0.3)]

len_alp = len(alp)  # length of alphabet
evidence_names = ['LM', 'Eps']  # types of evidences during simulation

# list_pre_phrase, list_target_phrase sizes need to match
list_pre_phrase = ['BOOK', 'IT_OC']  # what user have typed already
list_target_phrase = ['BOOKC', 'IT_OCC']  # what user wants to type
backspace_prob = 1. / 100.  # probability of backspace at each sequence

# Flag parameters
lm_flag = 0  # 1 if language model is active
save_flag = 1  # 1 if findings will be saved afterwards
one_cycle_flag = 1  # 1 if system terminates after 1 sequence only

max_num_mc = 100  # maximum number of monte carlo simulations
delta = 5.  # amount of shift for positive distribution
alpha = 0.05
sc_threshold = 10000  # remove redundant samples

conjugator = EvidenceFusion(evidence_names, len_dist=len_alp)
seq_stat_holder, acc_stat_holder, auc_stat_holder, itr_holder = [], [], [], []
alphabet = list(string.ascii_uppercase) + ['-'] + ['<']

# initialize bar-graph and length of iterations
len_bar = max_num_mc * len(list_target_phrase) * len(query_methods)
progress_bar(0, len_bar, prefix='Progress:', suffix='Complete', length=50)
bar_counter = 0

index_user = 0
# accuracy, sequence_spend lists for the application
# all will have the shape [num_users x num_alpha x num_mc_samples]
acc_overall, seq_overall, target_posterior_overall, itr_overall = [], [], [], []

# load filename and initialize the user from MATLAB file
tmp = sio.loadmat(path)
x = tmp['scores']
y = tmp['trialTargetness']
auc_user = tmp['meanAuc'][0][0]
auc_stat_holder.append(auc_user)

# modify data for outliers
y = y[x > -sc_threshold]
x = x[x > -sc_threshold]
y = y[x < sc_threshold]
x = x[x < sc_threshold]
x[y == 1] += delta

# create the oracle for the user
oracle = BinaryRSVPOracle(x, y, phrase='A', alp=alp)

# create an EEG model that fits   the user explicitly
bandwidth = 1.06 * min(np.std(x),
                       iqr(x) / 1.34) * np.power(x.shape[0], -0.2)
classes = np.unique(y)
cls_dep_x = [x[np.where(y == classes[i])] for i in range(len(classes))]
dist = []  # distributions for positive and negative classes
for i in range(len(classes)):
    dist.append(KernelDensity(bandwidth=bandwidth))

    dat = np.expand_dims(cls_dep_x[i], axis=1)
    dist[i].fit(dat)

# iterate over given phrases
for idx_phrase in range(len(list_pre_phrase)):
    # initialize pre-phrase and phrase of the iteration
    pre_phrase = alp[np.random.randint(len(alp))]
    phrase = pre_phrase + alp[np.random.randint(len(alp))]
    oracle.phrase = phrase

    # create the oracle and the decision maker.
    decision_maker = DecisionMaker(state='', len_query=len_query, alphabet=alp,
                                   max_num_seq=max_num_seq,
                                   min_num_seq=min_num_seq)

    # arrays to hold sequence and accuracy information
    seq_holder, acc_holder = [], []
    # have shapes [num_alpha x num_mc_samples], [num_mc_samples]
    target_dist = [[] for idx_duplicate in range(2)]
    target_dif_entropy = [[] for idx_duplicate in range(2)]
    target_dif_momentum = [[] for idx_duplicate in range(2)]
    list_itr = [[] for idx_duplicate in range(2)]

    list_letter = [[] for idx_duplicate in range(2)]
    list_prob = [[] for idx_duplicate in range(2)]

    query_count = 0
    for query in query_methods:

        decision_maker.query_method = query

        # for alpha in list_alpha:
        alpha_count = 0

        # Adjust maximum and minimum number of sequences before decision
        decision_maker.min_num_seq = min_num_seq
        decision_maker.max_num_seq = max_num_seq

        # ser momentum hyper-parameter for decision
        decision_maker.momentum_alpha = alpha * 2.6

        entropy_holder, seq_elapsed = [], []
        correct_sel = 0
        for idx in range(max_num_mc):
            progress_bar(bar_counter + 1, len_bar, prefix='Progress:',
                         suffix='Complete', length=50)

            decision_maker.reset(state=pre_phrase)
            oracle.reset()
            oracle.update_state(decision_maker.displayed_state)

            seq_till_correct = [0] * (len(phrase) - len(pre_phrase))
            d_counter = 0
            while decision_maker.displayed_state != phrase:

                # update displayed state from the user
                tmp_displayed_state = "_" + \
                                      decision_maker.displayed_state

                lm_prior = np.abs(np.random.randn(len(alp)))
                lm_prior[alp.index(oracle.state)] /= 100
                lm_prior /= np.sum(lm_prior)

                # lm_prior[alp.index(phrase)] = np.power(.1, 3)
                # lm_prior = lm_prior / np.sum(lm_prior)

                prob = conjugator.update_and_fuse({'LM': lm_prior})
                prob_new = np.array([i for i in prob])
                d, stimuli = decision_maker.decide(prob_new)

                sti = stimuli['stimuli'][0][0][1:]
                list_letter[alpha_count].append([sti])
                list_prob[alpha_count].append([prob_new])
                while True:
                    # get answers from the user
                    score = oracle.answer(sti)

                    # get the likelihoods for the scores
                    likelihood = []
                    for i in score:
                        dat = np.squeeze(i)
                        dens_0 = dist[0].score_samples(
                            dat.reshape(1, -1))[0]
                        dens_1 = dist[1].score_samples(
                            dat.reshape(1, -1))[0]
                        likelihood.append(np.asarray([dens_0, dens_1]))
                    likelihood = np.array(likelihood)
                    # compute likelihood ratio for the query
                    lr = np.exp(likelihood[:, 1] - likelihood[:, 0])

                    # initialize evidence with all ones
                    evidence = np.ones(len_alp)

                    c = 0
                    # update evidence of the queries that are asked
                    for q in sti:
                        idx = alp.index(q)
                        evidence[idx] = lr[c]
                        c += 1

                    # update posterior and decide what to do
                    prob = conjugator.update_and_fuse({'Eps': evidence})
                    prob_new = np.array([i for i in prob])
                    d, stimuli = decision_maker.decide(prob_new)

                    list_prob[alpha_count].append([prob_new])
                    if d:
                        tmp_dist = np.array(
                            decision_maker.list_epoch[-2][
                                'list_distribution'])
                        tmp_post = np.mean(tmp_dist, axis=0)
                        itr = (np.log2(len(alp)) + tmp_post * np.log2(
                            tmp_post) + (1 - tmp_post) * np.log2(
                            (np.maximum(eps, 1 - tmp_post)) / (
                                    len_alp - 1)))
                        target_dist[alpha_count].append(
                            list(tmp_dist[:, alp.index(oracle.state)]))

                        ent = (-1) * tmp_dist * np.log2(tmp_dist)
                        ent = np.sum(ent, axis=1)
                        diff_ent = - ent[1:] + ent[:-1]

                        diff_m = tmp_dist[:-1] * (np.log(tmp_dist[1:]) -
                                                  np.log(tmp_dist[:-1]))
                        diff_m = np.sum(diff_m, axis=1)

                        target_dif_entropy[alpha_count].append(diff_ent)
                        target_dif_momentum[alpha_count].append(diff_m)
                        list_itr[alpha_count].append(itr)

                        break

                    sti = stimuli['stimuli'][0][0][1:]
                    list_letter[alpha_count].append([sti])


                    # after decision update the user about the current delta
                    oracle.update_state(decision_maker.displayed_state)


                # Reset the conjugator before starting a new epoch
                # Starts the new task with a clear history
                conjugator.reset_history()
                seq_till_correct[d_counter] += len(
                    decision_maker.list_epoch[-2]['list_sti'])
                if (decision_maker.list_epoch[-2]['decision'] == phrase[
                    len(pre_phrase) + d_counter] and
                        decision_maker.displayed_state == phrase[0:len(
                            decision_maker.displayed_state)]):
                    correct_sel += 1
                    d_counter += 1

                # if number of sequences is fixed then hold time series
                if max_num_seq == min_num_seq:
                    asd = 1

                if one_cycle_flag:
                    break

            # store number of sequences spent for correct decision
            seq_elapsed.append(seq_till_correct[0])
            # one of the iterations is completed! update wait-bar
            bar_counter += 1

        seq_holder.append(seq_elapsed)
        acc_holder.append(1. * correct_sel / max_num_mc)

        alpha_count += 1
        query_count += 1

target_posterior_overall.append(target_dist)
itr_overall.append(list_itr)
seq_overall.append(seq_holder)
acc_overall.append(acc_holder)
index_user += 1

tmp = np.array(seq_overall)
tmp = np.swapaxes(tmp, 0, 1)
tmp = np.mean(tmp, 1)
tmp = np.mean(tmp, 1)
