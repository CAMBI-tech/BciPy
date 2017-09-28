""" Supports only Matlab offline analysis functionality. Specifically
designed to return required values in required format (trialwise) for 
RSVPKeyboard"""

import numpy as np


def trigger_decoder(x, trigger_partitioner):
    """ Decodes RSVPKeyboard stored data using the trigger partitioner
        Args:
            x(matlab_array): Nx1 array denoting triggers
            trigger_partitioner(dictionary): Stores trigger information.
            struct data type from MATLAB is converted into dictionary in py.
        Returns: A (list) of elements
            label_seq(ndarray)" num_seq x num_trial array with trial triggers
            timing_seq(ndarray): num_seq x num_trial array with trial time idx
            true_seq(ndarray): num_seq x 1  array with target letter for seq
            target_seq(ndarray): num_seq x num_trial array with targetness
    """

    len_win = trigger_partitioner['windowLengthinSamples']
    id_sequence_end = trigger_partitioner['sequenceEndID']
    id_pause = trigger_partitioner['pauseID']
    id_fixation = trigger_partitioner['fixationID']
    offset_trig = trigger_partitioner['TARGET_TRIGGER_OFFSET']

    # Find the falling edges and labels of the trials
    x = np.asarray(x)
    x = x.reshape(1, len(x))[0]
    x = x.astype(np.int32)
    time_last = len(x) - len_win
    x = x[0:time_last]
    time_fall_edge = np.where(np.diff(x) < 0)[0] - 1
    labels_x = x[time_fall_edge]

    # TODO: check for paused sequences
    # If pause id located, all sequence information should be discarded
    # prior to the pause. (All data or this sequence?)

    # Calculate number of sequences
    # Fixation denotes the sequence start, sequence end ID denotes end
    num_finished_seq = np.sum(labels_x == id_sequence_end)
    idx_fix = np.array(np.where(labels_x == id_fixation))[0]
    idx_end_seq = np.array(np.where(labels_x == id_sequence_end))[0]
    assert len(idx_fix) == len(idx_end_seq), 'Sequence start - end number ' \
                                             'mismatch.'

    # Decompose labels into sequences
    labels_seq, true_seq, target_seq, timing_seq = [], [], [], []
    for i in range(len(idx_end_seq)):
        labels_seq.append(labels_x[idx_fix[i] + 1:idx_end_seq[i]])
        timing_seq.append(time_fall_edge[idx_fix[i] + 1:idx_end_seq[i]])

    # Number different triggers in a sequence
    # TODO: check length of completed sequences
    len_sequence = len(labels_x) / num_finished_seq

    # Check if the mode is calibration :
    # In calibration mode the target letter trigger is presented before the
    # fixation, which causes the first index to be allocated with this
    # trigger instead of the fixation. This relation can be used as a lever.
    if idx_fix[0] != 0:  # Check if calibration
        for i in range(len(idx_end_seq)):
            true_seq.append(labels_x[idx_fix[i] - 1] - offset_trig)
            target_seq.append((labels_seq[i] == true_seq[i]).astype(np.int32))
    else:  # Another task otherwise
        true_seq = [np.zeros(
            int(idx_end_seq[0] - (idx_fix[0] + 1)))] * num_finished_seq
        target_seq = [np.zeros(
            int(idx_end_seq[0] - (idx_fix[0] + 1)))] * num_finished_seq

    labels_seq = np.array(labels_seq)
    timing_seq = np.array(timing_seq)
    true_seq = np.array(true_seq)
    target_seq = np.array(target_seq)

    return [labels_seq, timing_seq, true_seq, target_seq]
