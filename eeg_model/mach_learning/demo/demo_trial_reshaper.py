import numpy as np
from eeg_model.mach_learning.trial_reshaper import trial_reshaper
from helpers.load import load_txt_data
from helpers.trigger_helpers import trigger_decoder


# A 3 channel dummy input
inp = np.array([range(4000)] * 3)
# Sampling frequency
fs = 256
# Downsampling ratio
k = 2


# Uncomment the calibration mode you want to test trial reshaper for below:
demo_mode = 'calibration'
# demo_mode = 'copy_phrase'
# demo_mode = 'free_spell'


# Load trigger file
trigger_data = trigger_decoder(trigger_loc=demo_mode + '.txt', mode=demo_mode)

# reshape function is applied to dummy data with given trigger file
arg = trial_reshaper(trial_target_info=trigger_data[1],
                     timing_info=trigger_data[2], filtered_eeg=inp, fs=256, k=2, mode=demo_mode)

# Print results.
print 'Reshaped trials:\n', arg[0], '\nLabels:', arg[1], '\nTotal number of sequences:', \
    arg[2], '\nTrial number in each sequence:', arg[3]
