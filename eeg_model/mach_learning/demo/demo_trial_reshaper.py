import numpy as np
from eeg_model.mach_learning.trial_reshaper import trial_reshaper

#give location of trigger file
loc = 'Path\\to\\triggers.txt'

# A 3 channel dummy input
inp = np.array([range(4000)]*3)
fs = 256
k = 2


outp = trial_reshaper(trigger_location = loc, filtered_eeg = inp, fs = 256, k = 2)

# Every symbol shown in a sequence is a trial, fixation marks are removed
first_trial_data = outp[1]
second_trial_data = outp[2]

first_trial_symbol = first_trial_data[0]
first_trial_explanation = first_trial_data[1]
first_trial_matrix = first_trial_data[2]

print 'first_trial_symbol:', first_trial_symbol
print
print 'first_trial_explanation:', first_trial_explanation
print
print 'first_trial_matrix:'
print first_trial_matrix