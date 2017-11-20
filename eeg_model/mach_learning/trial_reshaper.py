import numpy as np
from helpers.trigger_helpers import trigger_decoder


def trial_reshaper(trial_target_info, timing_info, filtered_eeg, fs, k, mode='calibration'):
    """

    :param trial_target_info: A list of strings which can take values like
        'target', 'nontarget', 'first_press_target', or other values for free spelling.

    :param timing_info: Trigger timings for each trial, a list of floats

    :param filtered_eeg: channel wise band pass filtered and down-sampled eeg data
        with format: [channel x signal_length]

    :param fs: sampling frequency

    :param k: down-sampling rate applied to the filtered eeg signal.

    :param mode: Operating mode, can be calibration free-spelling, copy phrase, etc.

    :return (reshaped_trials, labels, num_of_sequences, trials_per_seq): Return type is a tuple.
    reshaped_trials =   3 dimensional np array first dimension is trials
                        second dimension is channels and third dimension is time samples.
    labels = np array for every trial's class.
    num_of_sequences = Integer for total sequence number, as written in trigger.txt
    trials_per_seq = np array which every i'th element is number of trials in i'th sequence
        where i = 0, .. , num_of_sequences - 1

    """

    if mode == 'calibration':
        trials_per_seq = []
        count = 0
        for symbol_info in trial_target_info:
            if symbol_info == 'first_pres_target':
                trials_per_seq.append(count)
                count = 0
            elif symbol_info == 'nontarget' or 'target':
                count += 1
            else:
                print('trial_reshaper does not understand triggers.txt files second column.')

        trials_per_seq = trials_per_seq[1:]
        trials_per_seq.append(count)
        trials_per_seq = np.array(trials_per_seq)

        for symbol_info_index in range(len(trial_target_info)):
            if trial_target_info[symbol_info_index] == 'first_pres_target':
                timing_info[symbol_info_index] = -1

        trial_target_info = filter(lambda x: x != 'first_pres_target', trial_target_info)
        timing_info = filter(lambda x: x != -1, timing_info)

        # triggers in seconds are mapped to triggers in number of samples. -1 is for indexing
        triggers = map(lambda x: int(x * fs / k) - 1, timing_info)

        # Number of samples in half a second that we are interested in:
        num_samples = int(1. * fs / 2 / k)

        # 3 dimensional np array first dimension is channels
        # second dimension is trials and third dimension is time samples.
        reshaped_trials = np.zeros((len(filtered_eeg), len(triggers), num_samples))

        # Label for every trial
        labels = np.zeros(len(triggers))

        # For every trial
        for trial in range(len(triggers)):
            if trial_target_info[trial] == 'target':
                labels[trial] = 1

            # For every channel
            for channel in range(len(filtered_eeg)):
                reshaped_trials[channel][trial] = \
                    filtered_eeg[channel][triggers[trial]:triggers[trial] + num_samples]

        num_of_sequences = int(sum(labels))

        return reshaped_trials, labels, num_of_sequences, trials_per_seq
    else:
        pass # This case has to be handled.
