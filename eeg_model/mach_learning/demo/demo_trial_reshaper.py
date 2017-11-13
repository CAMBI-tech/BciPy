import numpy as np
from eeg_model.mach_learning.trial_reshaper import trial_reshaper
from helpers.load import load_txt_data


# A 3 channel dummy input
inp = np.array([range(4000)] * 3)
fs = 256
k = 2

trigger_loc = load_txt_data()

arg = trial_reshaper(trigger_loc=trigger_loc, filtered_eeg=inp, fs=256, k=2)

print 'Reshaped trials:'
print arg[0]
print
print 'Labels:'
print arg[1]
