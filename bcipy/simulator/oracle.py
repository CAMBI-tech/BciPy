import numpy as np
from scipy.stats import norm, iqr
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import auc, roc_curve
import string
from bcipy.helpers.load import read_data_csv, load_experimental_data, \
    load_json_parameters
from bcipy.signal_processing.sig_pro import sig_pro
import numpy as np
import argparse
import random
from bcipy.helpers.triggers import trigger_decoder


class BinaryRSVPOracle(object):
    """ RSVPKeyboard oracle implementation (interpreted as user)
        Attr:
            phrase(str): user's intention to write (can be anything / sentence,word etc.)
            state(char): at a particular time, what letter user wants to type
            dist(list[kernel_density]): distribution estimates for different classes.
            alp(list[char]): alphabet. final symbol should be the erase symbol
            auc(float): area under the ROC curve of the user ()
        """

    def __init__(self, x, y, phrase, alp=list(string.ascii_uppercase) + ['_'] + ['<']):
        """ Args:
                x(ndarray[float]): scores of trial samples
                y(ndarray[int]): label values of trial samples (1 is for positive)
                phrase(str): phrase in users ming (e.g.: I_love_cookies)
            """
        self.alp = alp
        self.phrase = phrase
        self.state = self.phrase[0]

        self.dist = form_kde_densities(x, y)
        fpr, tpr, thresholds = roc_curve(y, x, pos_label=1)
        self.auc = auc(fpr, tpr)

    def reset(self):
        self.state = self.phrase[0]

    def update_state(self, delta):
        """ update the oracle state based on the displayed state(delta) and the phrase.
            Args:
                delta(str): what is typed on the screen
         """
        if self.phrase == delta:
            None
        else:
            if self.phrase[0:len(delta)] == delta:
                self.state = self.phrase[len(delta)]
            else:
                self.state = '<'

    def answer(self, q):
        """ binary oracle responds based on the query/state match
            Args:
                q(list[char]): stimuli flashed on screen
            Return:
                sc(ndarray[float]): scores sampled from different distributions
                    this is the evidence
                """

        # label the query so we can sample from 0 and 1
        label = [int(q[i] == self.state) for i in range(len(q))]
        sc = []
        for i in label:
            sc.append(self.dist[i].sample(1))

        return np.asarray(sc)


class SequenceRSVPOracle(object):
    """ RSVPKeyboard oracle implementation (interpreted as user)
        Attr:
            phrase(str): user's intention to write (can be anything / sentence,word etc.)
            state(char): at a particular time, what letter user wants to type
            alp(list[char]): alphabet. final symbol should be the erase symbol
        """

    def __init__(self, data_folder, phrase, alp=list(string.ascii_uppercase) + ['_'] + ['<']):
        """ Args:
                x(ndarray[float]): scores of trial samples
                y(ndarray[int]): label values of trial samples (1 is for positive)
                phrase(str): phrase in users ming (e.g.: I_love_cookies)
            """
        self.alp = alp
        self.phrase = phrase
        self.state = self.phrase[0]

        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_folder', default=None)
        parser.add_argument('-p', '--parameters_file',
                            default='D:/BCIpy/BciPy/bcipy/parameters/parameters.json')
        args = parser.parse_args()
        parameters = load_json_parameters(args.parameters_file, value_cast=True)
        raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
            data_folder + '/' + parameters.get('raw_data_name', 'raw_data.csv'))

        downsample_rate = parameters.get('down_sampling_rate', 2)
        self.trigger_holder = None
        self.filtered_data = sig_pro(raw_dat, fs=fs, k=downsample_rate)
        triggers_file = parameters.get('triggers_file_name', 'triggers.txt')
        self.form_trigger_list(data_folder, triggers_file)

    def reset(self):
        self.state = self.phrase[0]

    def update_state(self, delta):
        """ update the oracle state based on the displayed state(delta) and the phrase.
            Args:
                delta(str): what is typed on the screen
         """
        if self.phrase == delta:
            None
        else:
            if self.phrase[0:len(delta)] == delta:
                self.state = self.phrase[len(delta)]
            else:
                self.state = '<'

    def answer(self, q):
        """ binary oracle responds based on the query/state match
            Args:
                q(list[char]): stimuli flashed on screen
            Return:
                triggers(list[float]): positions for the sequence
                    this is the evidence
                """

        # label the query so we can sample from 0 and 1
        label = [int(q[i] == self.state) for i in range(len(q))]
        label = np.array(label)

        if np.sum(label == 1) == 0:
            trig = self.trigger_holder[0][0]
            trig[0] = self.trigger_holder[1][0][0]
        else:
            loc = np.where(label == 1)[0]
            trig = random.sample(self.trigger_holder[int(loc)], 1)[0]

        return trig

    def form_trigger_list(self, data_folder, triggers_file):

        _, t_t_i, t_i, offset = trigger_decoder(mode='calibration', trigger_loc=f"{data_folder}/{triggers_file}")
        start_idx = np.where([a == 'first_pres_target' for a in t_t_i])[0]
        target_idx = np.where([a == 'target' for a in t_t_i])[0]
        locations = target_idx - start_idx - 1

        self.trigger_holder = [[]] * (np.max(locations) + 1)
        for idx in range(np.max(locations) + 1):
            tmp = []
            for idx_2 in list(list(np.where(locations == idx)[0])):
                tmp.append(list(t_i[start_idx[idx_2] + 1:start_idx[idx_2] + 11]))

            self.trigger_holder[idx] = tmp

        # with open(trigger_loc, 'r') as text_file:
        #     lines = [line.split() for line in text_file]
        #
        # targets, triggers, tar_info, tri_info = [], [], [], []
        # for line in lines:
        #     if 'calibration_trigger' not in line:
        #         if 'first_pres_target' not in line:
        #             if 'fixation' in line:
        #                 targets.append(tar_info)
        #                 triggers.append(tri_info)
        #
        #                 tar_info = []
        #                 tri_info = []
        #             else:
        #                 tar_info.append(int(line[1] == 'target'))
        #                 tri_info.append(float(line[2]))
        #
        # targets = np.array(targets[1::])
        # triggers = np.array(triggers[1::])
        # locations = np.where(targets == 1)[1]
        #
        # self.trigger_holder = [[]] * (np.max(locations) + 1)
        # for idx in range(np.max(locations)):
        #     tmp = []
        #     for idx_2 in list(list(np.where(locations == idx)[0])):
        #         tmp.append(list(triggers[idx_2]))
        #
        #     self.trigger_holder[idx] = tmp


def form_kde_densities(x, y):
    """ Fits kernel (gaussian) density estimates to the data
        Args:
            x(ndarray[float]): samples
            y(ndarray[labels]): labels
        Return:
            dist(list[kernel_density]): distributions. number of distributions is
                equal to the number of unique elements in y vector.
            """
    bandwidth = 1.06 * min(np.std(x),
                           iqr(x) / 1.34) * np.power(x.shape[0], -0.2)
    classes = np.unique(y)
    cls_dep_x = [x[np.where(y == classes[i])] for i in range(len(classes))]
    dist = []
    for i in range(len(classes)):
        dist.append(KernelDensity(bandwidth=bandwidth))

        dat = np.expand_dims(cls_dep_x[i], axis=1)
        dist[i].fit(dat)

    return dist
