import numpy as np


def trial_reshaper(trial_target_info, timing_info, filtered_eeg, fs, k, mode, offset=0,
        channel_map=(1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0)):
    """

    :param trial_target_info: A list of strings which can take values:
        'target', 'nontarget', 'first_press_target'

    :param timing_info: Trigger timings for each trial, a list of floats

    :param filtered_eeg: channel wise band pass filtered and down-sampled data
        with format: [channel x signal_length]

    :param fs: sampling frequency

    :param k: down-sampling rate applied to the filtered eeg signal.

    :param mode: Operating mode, can be 'calibration', 'copy_phrase', etc.

      :param channel_map: A binary list, if i'th element is 0, i'th channel
        in filtered_eeg is removed.

    :return (reshaped_trials, labels, num_of_sequences, trials_per_seq): Return
         type is a tuple.
    reshaped_trials =   3 dimensional np array first dimension is channels
                        second dimension is trials and third dimension is time
                        samples.
    labels = np array for every trial's class.
    num_of_sequences = Integer for total sequence number, as written in
        trigger.txt
    trials_per_seq = number of trials in each sequence
    offset: the calculated offset of triggers. To be subtracted from data.
        It will return a negative value if it needs to be added.

    """

    # Remove the channels that we are not interested in
    channel_indexes_to_remove = []
    for channel_index in range(len(filtered_eeg)):
        if channel_map[channel_index] == 0:
            channel_indexes_to_remove.append(channel_index)

    filtered_eeg = np.delete(filtered_eeg,
                             channel_indexes_to_remove, axis=0)

    # Number of samples in half a second that we are interested in
    num_samples = int(1. * fs / 2 / k)

    if mode == 'calibration':
        trials_per_seq = []
        count = 0
        # Count every sequences trials
        for symbol_info in trial_target_info:
            if symbol_info == 'first_pres_target':
                trials_per_seq.append(count)
                count = 0
            elif symbol_info == 'nontarget' or 'target':
                count += 1
            else:
                raise Exception('Incorrectly formatted trigger file. \
                See trial_reshaper documentation for expected input.')

        # The first element is garbage. Get rid of it.
        trials_per_seq = trials_per_seq[1:]
        # Append the last sequences trial number
        trials_per_seq.append(count)
        # Make the list a numpy array.
        trials_per_seq = np.array(trials_per_seq)

        # Mark every element in timing_info if 'first_pres_target'
        for symbol_info_index in range(len(trial_target_info)):
            if trial_target_info[symbol_info_index] == 'first_pres_target':
                timing_info[symbol_info_index] = -1

        # Get rid of 'first_pres_target' trials information
        trial_target_info = filter(lambda x: x != 'first_pres_target',
                                   trial_target_info)
        timing_info = filter(lambda x: x != -1, timing_info)

        # triggers in seconds are mapped to triggers in number of samples.
        triggers = map(lambda x: int((x - offset) *fs / k), timing_info)

        # 3 dimensional np array first dimension is channels
        # second dimension is trials and third dimension is time samples.
        reshaped_trials = np.zeros(
            (len(filtered_eeg), len(triggers), num_samples))

        # Label for every trial
        labels = np.zeros(len(triggers))

        for trial in range(len(triggers)):
            # Assign targetness to labels for each trial
            if trial_target_info[trial] == 'target':
                labels[trial] = 1

            # if (np.abs(filtered_eeg[13][triggers[trial]:triggers[trial] + num_samples]) > 500).any():
            #     print 'artifact at: {}'.format(trial)

            # For every channel append filtered channel data to trials
            for channel in range(len(filtered_eeg)):
                reshaped_trials[channel][trial] = \
                    filtered_eeg[channel][
                    triggers[trial]:triggers[trial] + num_samples]

        num_of_sequences = int(sum(labels))

        return reshaped_trials, labels, num_of_sequences, trials_per_seq

    elif mode == 'copy_phrase':

        # triggers in seconds are mapped to triggers in number of samples.
        triggers = map(lambda x: int((x - offset) *fs / k), timing_info)

        # 3 dimensional np array first dimension is channels
        # second dimension is trials and third dimension is time samples.
        reshaped_trials = np.zeros(
            (len(filtered_eeg), len(triggers), num_samples))

        # Label for every trial
        labels = np.zeros(len(triggers))

        for trial in range(len(triggers)):
            # Assign targetness to labels for each trial
            if trial_target_info[trial] == 'target':
                labels[trial] = 1

            # For every channel append filtered channel data to trials
            for channel in range(len(filtered_eeg)):
                reshaped_trials[channel][trial] = \
                    filtered_eeg[channel][
                    triggers[trial]:triggers[trial] + num_samples]

        # In copy phrase, num of sequence is assumed to be 1.
        num_of_sequences = 1
        # Since there is only one sequence, all trials are in the sequence
        trials_per_seq = len(triggers)

        return reshaped_trials, labels, num_of_sequences, trials_per_seq

    elif mode == 'free_spell':

        # triggers in seconds are mapped to triggers in number of samples.
        triggers = map(lambda x: int((x - offset) *fs / k), timing_info)

        # 3 dimensional np array first dimension is channels
        # second dimension is trials and third dimension is time samples.
        reshaped_trials = np.zeros(
            (len(filtered_eeg), len(triggers), num_samples))

        labels = None

        for trial in range(len(triggers)):

            # For every channel append filtered channel data to trials
            for channel in range(len(filtered_eeg)):
                reshaped_trials[channel][trial] = \
                    filtered_eeg[channel][
                    triggers[trial]:triggers[trial] + num_samples]

        # In copy phrase, num of sequence is assumed to be 1.
        num_of_sequences = 1
        # Since there is only one sequence, all trials are in the sequence
        trials_per_seq = len(triggers)

        return reshaped_trials, labels, num_of_sequences, trials_per_seq
    else:
        raise Exception('Trial_reshaper does not work in this operating mode.')
