import numpy as np
from eeg_model.mach_learning.trial_reshaper import trial_reshaper

# A 3 channel dummy input
inp = np.array([range(4000)]*3)
fs = 256
k = 2

outp = trial_reshaper(trigger_location = None, filtered_eeg = inp, fs = 256, k = 2)

print 'Reshaped trials:'
print outp[0]
print
print 'Labels:'
print outp[1]
