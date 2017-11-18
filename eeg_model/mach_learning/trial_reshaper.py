import numpy as np
from helpers.trigger_helpers import trigger_decoder


def trial_reshaper(trigger_data, filtered_eeg, fs, k, mode = 'calibration'):
    """

    :param trigger_data: Output of trigger_helpers.trigger_decoder
    :param filtered_eeg: channel wise band pass filtered and down-sampled eeg data with format: [channel x signal_length]
    :param fs: sampling frequency
    :param k: down-sampling rate applied to the filtered eeg signal.

    :return (reshaped_trials, labels, num_of_sequences, total_trials_in_seq): Return type is a tuple.
    reshaped_trials =   3 dimensional np array first dimension is trials
                        second dimension is channels and third dimension is time samples.
    labels = np array for every trial's class.
    num_of_sequences = Integer for total sequence number, as written in trigger.txt
    total_trials_in_seq = np array which every i'th element is number of trials in i'th sequence where i = 0, .. , num_of_sequences - 1

     """

    trigger_txt = trigger_data[1]
    trigger_timing = trigger_data[2]

    if mode == 'calibration':
        total_trials_in_seq = []
        count = 0
        for symbol_info in trigger_txt:
            if symbol_info == 'first_pres_target':
                total_trials_in_seq.append(count)
                count = 0
            elif symbol_info == 'nontarget' or 'target':
                count += 1
            else:
                print('trial_reshaper does not understand triggers.txt files second column.')

        total_trials_in_seq = total_trials_in_seq[1:]
        total_trials_in_seq.append(count)
        total_trials_in_seq = np.array(total_trials_in_seq)

        for symbol_info_index in range(len(trigger_txt)):
            if trigger_txt[symbol_info_index] == 'first_pres_target':
                trigger_timing[symbol_info_index] = '-1'

        trigger_txt = filter(lambda x: x != 'first_pres_target', trigger_txt)
        trigger_timing = filter(lambda x: x != -1, map(eval, trigger_timing))

        # triggers in seconds are mapped to triggers in number of samples. -1 is for indexing
        triggers = map(lambda x: int(x * fs / k) - 1, trigger_timing)

        # Number of samples in half a second that we are interested in:
        num_samples = int(1. * fs / 2 / k)

        # 3 dimensional np array first dimension is channels
        # second dimension is trials and third dimension is time samples.
        reshaped_trials = np.zeros((len(filtered_eeg), len(triggers), num_samples))

        # Label for every trial
        labels = np.zeros(len(triggers))

        # For every trial
        for trial in range(len(triggers)):
            if trigger_txt[trial] == 'target':
                labels[trial] = 1

            # For every channel
            for channel in range(len(filtered_eeg)):
                reshaped_trials[channel][trial] = filtered_eeg[channel][triggers[trial]:triggers[trial] + num_samples]

        num_of_sequences = int(sum(labels))

        return reshaped_trials, labels, num_of_sequences, total_trials_in_seq
    else:
        pass # This case has to be handled.
