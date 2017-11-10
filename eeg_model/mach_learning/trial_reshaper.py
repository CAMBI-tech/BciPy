import numpy as np
from helpers.load import load_txt_data


def trial_reshaper(trigger_loc, filtered_eeg, fs, k):
    """

    :param trigger_loc: location of the trigger.txt file to read triggers from
    :param filtered_eeg: channel wise band pass filtered and downsampled eeg data with format: [channel x signal_length]
    :param fs: sampling frequency
    :param k: downsampling rate applied to the filtered eeg signal.

    :return [reshaped_trials, labels]: Return type is a list.
    reshaped_trials =   3 dimensional np array first dimension is trials
                        second dimension is channels and third dimension is time samples.
    labels = np array for every trial's class.

     """

    # Load triggers.txt
    if not trigger_loc:
        trigger_loc = load_txt_data()

    with open(trigger_loc, 'r') as text_file:
        trigger_txt = [line.replace('\n', '').split() for line in text_file
                       if
                       'fixation' not in line and 'first_pres_target' not in line]

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
            reshaped_trials[channel][trial] = filtered_eeg[channel][
                                              triggers[trial]:triggers[
                                                                  trial] + num_samples]

    return [reshaped_trials, labels]
