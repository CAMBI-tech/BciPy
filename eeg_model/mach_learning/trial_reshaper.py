import numpy as np
from helpers.trigger_helpers import trigger_decoder


def trial_reshaper(trigger_loc, filtered_eeg, fs, k):
    """

    :param trigger_loc: location of the trigger.txt file to read triggers from
    :param filtered_eeg: channel wise band pass filtered and down-sampled eeg data with format: [channel x signal_length]
    :param fs: sampling frequency
    :param k: down-sampling rate applied to the filtered eeg signal.

    :return (reshaped_trials, labels, num_of_sequences): Return type is a tuple.
    reshaped_trials =   3 dimensional np array first dimension is trials
                        second dimension is channels and third dimension is time samples.
    labels = np array for every trial's class.

     """

    trigger_txt = trigger_decoder(trigger_loc=trigger_loc)

    # Every trial's trigger timing
    triggers = [eval(line[2]) for line in trigger_txt]

    # triggers in seconds are mapped to triggers in number of samples. -1 is for indexing
    triggers = map(lambda x: int(x * fs / k) - 1, triggers)

    # Number of samples in half a second that we are interested in:
    num_samples = int(1. * fs / 2 / k)

    # 3 dimensional np array first dimension is channels
    # second dimension is trials and third dimension is time samples.
    reshaped_trials = np.zeros((len(filtered_eeg), len(triggers), num_samples))

    # Label for every trial
    labels = np.zeros(len(triggers))

    # For every trial
    for trial in range(len(triggers)):
        if trigger_txt[trial][1] == 'target':
            labels[trial] = 1

        # For every channel
        for channel in range(len(filtered_eeg)):
            reshaped_trials[channel][trial] = filtered_eeg[channel][triggers[trial]:triggers[trial] + num_samples]


    num_of_sequences = np.sum(labels)

    return reshaped_trials, labels, num_of_sequences
