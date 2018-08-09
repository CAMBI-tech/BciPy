from bcipy.simulator.oracle import SequenceRSVPOracle
import scipy.io as sio
from bcipy.bci_tasks.main_frame import DecisionMaker, EvidenceFusion
from scipy.stats import iqr
from sklearn.neighbors.kde import KernelDensity
from bcipy.helpers.bci_task_related import trial_reshaper
from bcipy.helpers.acquisition_related import analysis_channels
from bcipy.helpers.load import read_data_csv, load_experimental_data, load_json_parameters
import string
import numpy as np
import argparse
from bcipy.helpers.load import load_classifier
from bcipy.helpers.triggers import trigger_decoder


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

alp = list(string.ascii_uppercase) + ['_'] + ['<']
len_alp = len(alp)
evidence_names = ['LM', 'Eps']

# phrase that is already typed before the task
pre_phrase = 'CA'
# target of the task
phrase = "CAESAR"
backspace_prob = 1. / 35.
min_num_seq = 1
max_num_seq = 15
max_num_mc = 10
data_folder = 'C:\\Users\\berkan\\Desktop\\Model\\'
model_folder = 'C:\\Users\\berkan\\Desktop\\Model\\'

model = load_classifier(filename=model_folder + 'model.pkl')

conjugator = EvidenceFusion(evidence_names, len_dist=len_alp)
decision_maker = DecisionMaker(state='', alphabet=alp)
decision_maker.min_num_seq = min_num_seq
decision_maker.max_num_seq = max_num_seq

oracle = SequenceRSVPOracle(data_folder=data_folder, phrase=phrase, alp=alp)
data = oracle.filtered_data

# number of sequences spent for correct decision (adds incorrect decisions)
seq_holder = [[], []]

target_dist = [[], []]

bar_counter = 0
method_count = 0

parameters = {}
sum_seq_till_correct = np.zeros(len(phrase))
downsample_rate = parameters.get('down_sampling_rate', 2)
mode = 'copy_phrase'
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_folder', default=None)
parser.add_argument('-p', '--parameters_file',
                    default='C:\\Users\\berkan\\Desktop\\GitProjects\\BciPy\\bcipy\\parameters\\parameters.json')
triggers_file = parameters.get('triggers_file_name', 'triggers.txt')
args = parser.parse_args()
parameters = load_json_parameters(args.parameters_file, value_cast=True)
raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
    data_folder + '/' + parameters.get('raw_data_name', 'raw_data.csv'))
_, t_t_i, t_i, offset = trigger_decoder(mode=mode, trigger_loc=f"{data_folder}/{triggers_file}")

channel_map = analysis_channels(channels, type_amp)

# progress_bar(0, max_num_mc, prefix='Progress:', suffix='Complete', length=50)
for idx in range(max_num_mc):
    # progress_bar(bar_counter + 1, max_num_mc, prefix='Progress:', suffix='Complete', length=50)
    print('MC simulation no:{}'.format(idx + 1))
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
            t_i = np.array(oracle.answer(sti['stimuli'][0][0][1::]))
            t_t_i = ["nontarget"] * len(t_i)

            x, y, num_seq, _ = trial_reshaper(t_t_i, t_i, data,
                                              mode='copy_phrase', fs=fs, k=downsample_rate,
                                              offset=offset, channel_map=channel_map)

            score = model.transform(x)
            lr = np.exp(score[:, 1] - score[:, 0])

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

    print('State:{}'.format(decision_maker.state)),
    seq_holder[method_count].append(seq_till_correct)
    bar_counter += 1
