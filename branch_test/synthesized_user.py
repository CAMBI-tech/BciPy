import numpy as np
from scipy.stats import iqr
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import auc, roc_curve


class Synth:
    """ Synthesized user base class. For the analogy please check Fallout 4!
        Synthetic user that is designed to simulate a typing
        task utilizing either dummy data or real user data.
            Attr:
                phrase(str): user's intended state to write
                    (can be anything / sentence,word etc.)
                state(char): the letter of intention (the current letter to be
                    typed to progress towards the phrase)
                alp(list[char]): alphabet.
                erase_command(str): backspace character for the system """

    def __init__(self, alp, erase_command, phrase, **kwargs):
        """ Args:
                alp(list[str]): alphabet
                erase_command(str): backspace character, used if synth wants
                    to erase a previously committed decision.
                phrase(str): phrase in users mind (e.g.: I_love_cookies) """
        self.alp = alp
        self.erase_command = erase_command
        self.phrase = phrase

    def update_state(self, delta):
        """ update the synth state based on the displayed state(delta)
            and the phrase the synth is initiated with.
            Args:
                delta(str): what is typed on the screen
         """
        if self.phrase == delta:
            # TODO: include a termination response here!
            pass
        else:
            if self.phrase[0:len(delta)] == delta:

                self.state = self.phrase[len(delta)]
            else:
                self.state = self.erase_command

    def answer(self, q, **kwargs):
        """ binary synth responds based on the query/state match
            Args:
                q(list[char]): stimuli flashed on screen """
        pass


class BinaryGaussianSynth(Synth):
    """ A synthesized user implementation which utilizes Gaussian feature models
            Attr:
                phrase(str): user's intention to write (can be anything /
                    sentence,word etc.)
                state(char): current letter of intent by the user
                dist(list[kernel_density]): distribution estimates
                alp(list[char]): alphabet
                a_mean(ndarray[float]): 2x1 float array denoting the means
                a_std(ndarray[int]): 2x1 float array denoting the std
                erase_command(str): backspace character for the system
            """

    def __init__(self, alp, erase_command, phrase, a_mean, a_std):
        """ Args:
                a_mean(ndarray[float]): 2x1 float array denoting the means
                a_std(ndarray[int]): 2x1 float array denoting the std
                phrase(str): phrase in users mind (e.g.: I_love_cookies)
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

    def answer(self, q, type_q):
        """ binary synth responds based on the query/state match
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


class BinaryRSVPSynth(Synth):
    """ RSVPKeyboard synthesized user implementation (interpreted as user)
        Attr:
            phrase(str): user's intention to write (can be anything / sentence,
                word etc.)
            state(char): at a particular time, what letter user wants to type
            dist(list[kernel_density]): distribution estimates for different c
                lasses.
            alp(list[char]): alphabet. final symbol should be the erase symbol
            auc(float): area under the ROC curve of the user ()
            erase_command(str): backspace character for the system
        """

    def __init__(self, alp, erase_command, phrase, x, y):
        """ Args:
                x(ndarray[float]): scores of trial samples
                y(ndarray[int]): label values of trial samples (1 is
                    for positive)
                phrase(str): phrase in users mind (e.g.: I_love_cookies)
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

    def answer(self, q):
        """ binary synth responds based on the query/state match
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
            dist(list[kernel_density]): distributions. number of distributions
                is equal to the number of unique elements in y vector. """
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
