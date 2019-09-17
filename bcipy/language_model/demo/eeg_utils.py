import numpy as np
from random import shuffle, randint
import math
from bcipy.helpers.task import alphabet


class simulate_eeg:
    """
    Simulating EEG given target symbols
    """

    def __init__(self, path2eeg):
        self.abc = alphabet()
        self.samples = eegs(path2eeg)

    def simulate(self, history):
        eeg = []
        for target in history:
            eeg.append(generate_eeg(self.samples, target, self.abc))
        return eeg


def eegs(path):
    """load EEG simulations"""

    sample = 0
    num_sym = 0
    eeg_dict = {}
    a_sample = []
    for line in open(path).readlines():
        line = line.split()
        if line:
            a_sample.append(float(line[0]))
            num_sym += 1
        else:
            num_sym = 0
            a_sample = np.array(a_sample)
            # assuming it's log likelihood and not negative ll
            transformed_dist = [math.e**(prob) for prob in a_sample]
            total = sum(transformed_dist)
            normalized_dist = [(prob / total) for prob in transformed_dist]
            eeg_dict[sample] = [-math.log(prob) for prob in normalized_dist]
            sample += 1
            a_sample = []
    return eeg_dict


def generate_eeg(eeg, ch, syms):
    """generate according
    to target and non-target
    symbols a simulated
    EEG distribution"""

    # generate a tuple with all symbols (currently does not include "<")
    symsC = syms[:]
    idx = symsC.index(ch)
    del symsC[idx]
    shuffle(symsC)
    sample_id = randint(0, 999)
    sample = eeg[sample_id]
    dist = []
    dist.append((ch, sample[0]))
    for i in range(len(symsC)):
        dist.append((symsC[i], sample[i + 1]))
    # dist = sorted(dist, key=lambda symbol: symbol[1])
    dist = [dist[0]] + sorted(dist[1:], key=lambda symbol: symbol[1])
    return dist
