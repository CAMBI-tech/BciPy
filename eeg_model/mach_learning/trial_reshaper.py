import numpy as np


def trial_reshaper(trigger_location, filtered_eeg, fs, k):
    """

    :param trigger_location: location of the trigger.txt file to read triggers from
    :param filtered_eeg: channel wise band pass filtered and downsampled eeg data with format: [channel x signal_length]
    :param fs: sampling frequency
    :param k: downsampling rate applied to the filtered eeg signal.

    :return dict: {1: '<', 'first_pres_target', ndarray([Number of channels x .5 second samples for trial]), 2: ...}
    Size of the returned dictionary is the size of the trials (fixations not included) marked in the triggers.txt

    Important assumption: Samples should start aligned with trigger information. i.e. First sample corresponds to t = 0 w.r.t. triggers.

    """

    with open(trigger_location, 'r') as text_file:
        dict = [x.replace('\n','').split() for x in text_file if not 'fixation' in x]

    # Every trial's trigger in seconds
    triggers = [eval(x[2]) for x in dict]

    # triggers in seconds are mapped to triggers in number of samples. -1 is for indexing
    triggered_samples = map(lambda x: int(x*fs/k) - 1, triggers)

    # Number of samples in half a second that we are interested in:
    Ns = int(1.*fs/2/k)

    # For every trial
    for z in range(len(triggered_samples)):
        # For every channel
        dict[z][2] = []
        for zz in range(len(filtered_eeg)):
            dict[z][2].append(filtered_eeg[zz][triggered_samples[z]:triggered_samples[z]+Ns])

        dict[z][2] = np.array(dict[z][2])

    return dict
