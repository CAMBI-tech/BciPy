import numpy as np
from scipy.stats import norm, iqr
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import auc, roc_curve


# TODO: define the root object to allow new design

class BinaryGaussianOracle(object):
    """ RSVPKeyboard oracle implementation (interpreted as user)
            Attr:
                phrase(str): user's intention to write (can be anything / sentence,word etc.)
                state(char): at a particular time, what letter user wants to type
                dist(list[kernel_density]): distribution estimates for different classes.
                alp(list[char]): alphabet. final symbol should be the erase symbol
                auc(float): area under the ROC curve of the user ()
                erase_command(str): backspace character for the system
            """

    def __init__(self, a_mean, a_std, phrase, alp, erase_command):
        """ Args:
                a_mean(ndarray[float]): 2x1 float array denoting the means
                a_std(ndarray[int]): 2x1 float array denoting the std
                phrase(str): phrase in users ming (e.g.: I_love_cookies)
            """
        self.alp = alp
        self.erase_command = erase_command

        self.mean = a_mean
        self.std = a_std
        self.phrase = phrase
        self.state = self.phrase[0]
        self.auc = None

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
                self.state = self.erase_command

    def answer(self, q, type_q):
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
            if i == 1:
                sc.append(np.random.normal(self.mean[1], self.std[1], 1))
            else:
                sc.append(np.random.normal(self.mean[0], self.std[0], 1))

        return np.asarray(sc)


class BinaryRSVPOracle(object):
    """ RSVPKeyboard oracle implementation (interpreted as user)
        Attr:
            phrase(str): user's intention to write (can be anything / sentence,word etc.)
            state(char): at a particular time, what letter user wants to type
            dist(list[kernel_density]): distribution estimates for different classes.
            alp(list[char]): alphabet. final symbol should be the erase symbol
            auc(float): area under the ROC curve of the user ()
            erase_command(str): backspace character for the system
        """

    def __init__(self, x, y, phrase, alp, erase_command):
        """ Args:
                x(ndarray[float]): scores of trial samples
                y(ndarray[int]): label values of trial samples (1 is for positive)
                phrase(str): phrase in users ming (e.g.: I_love_cookies)
            """
        self.alp = alp
        self.phrase = phrase
        self.erase_command = erase_command

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
            pass
        else:
            if self.phrase[0:len(delta)] == delta:

                self.state = self.phrase[len(delta)]
            else:
                self.state = self.erase_command

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
