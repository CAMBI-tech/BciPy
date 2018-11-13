from bcipy.helpers.stimuli_generation import random_rsvp_calibration_seq_gen, best_case_rsvp_seq_gen
import numpy as np


def _demo_random_rsvp_sequence_generator():
    alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']

    num_samples = int(np.random.randint(10, 30, 1))
    len_samples = int(np.random.randint(1, 20, 1))

    print('Number of sequences:{}, Number of trials:{}'.format(num_samples,
                                                               len_samples))

    print('Alphabet:{}'.format(alp))
    schedule = random_rsvp_calibration_seq_gen(alp=alp,
                                               num_sti=num_samples,
                                               len_sti=len_samples)
    sequences = schedule[0]
    timing = schedule[1]
    color = schedule[2]

    for i in range(len(sequences)):
        print('seq{}:{}'.format(i, sequences[i]))
        if len(set(sequences[i][2::])) != len(sequences[i][2::]):
            raise Exception('Letter repetition!')
        print('time{}:{}'.format(i, timing[i]))
        print('color{}:{}'.format(i, color[i]))
    return 0


def _demo_best_case_sequence_generator():
    alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']

    num_samples = int(np.random.randint(1, 10, 1))
    len_samples = int(np.random.randint(1, 20, 1))

    print('Number of sequences:{}, Number of trials:{}'.format(num_samples,
                                                               len_samples))
    print('Alphabet:{}'.format(alp))
    for i in range(num_samples):
        p = np.random.randint(0, 10, len(alp))
        print('Probabilities:{}'.format(p))
        schedule = best_case_rsvp_seq_gen(alp=alp, session_stimuli=p, num_sti=1,
                                          len_sti=len_samples)
        sequence = schedule[0]
        timing = schedule[1]
        color = schedule[2]

        print('seq{}:{}'.format(i, sequence))
        print('time{}:{}'.format(i, timing))
        print('color{}:{}'.format(i, color))

    return 0

if __name__ == '__main__':
    _demo_best_case_sequence_generator()
    _demo_random_rsvp_sequence_generator()
