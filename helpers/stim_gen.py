""" timing and color information either hard coded or read from a parameters
file """
import numpy as np


def n_best_case_rsvp_seq_gen(alp, p, timing=[1, 0.2],
                             color=['red', 'white'], num_sti=1,
                             len_sti=10):
    """ generates RSVPKeyboard sequence by picking n-most likeliy letters.
        Args:
            alp(list[str]): alphabet (can be arbitrary)
            p(ndarray[float]): quantifier metric for query selection
                dim(p) = card(alp)!
            timing(list[float]): Task specific timing for generator
            color(list[str]): Task specific color for generator
                First element is the target, second element is the fixation
                Observe that [-1] element represents the trial information
            num_sti(int): number of random stimuli to be created
            len_sti(int): number of trials in a sequence
        Return:
            schedule_seq(tuple(
                samples[list[list[str]]]: list of sequences
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled sequences
            """

    len_alp = len(alp)
    if len_alp != len(p):
        raise Exception('Missing information about alphabet. len(alp):{}, '
                        'len(p):{}, should be same!'.format(len(alp), len(p)))

    idx = np.argsort(p)[::-1][0:len_sti]

    samples, times, colors = [], [], []
    for idx_num in range(num_sti):
        sample = ['+']
        idx = np.random.permutation(idx)
        sample += [alp[i] for i in idx]
        samples.append(sample)
        times.append([timing[i] for i in range(len(timing) - 1)] +
                     [timing[-1]] * len_sti)
        colors.append([color[i] for i in range(len(color) - 1)] +
                      [color[-1]] * len_sti)

    schedule_seq = (samples, times, colors)

    return schedule_seq


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
        schedule = n_best_case_rsvp_seq_gen(alp=alp, p=p, num_sti=1,
                                            len_sti=len_samples)
        sequence = schedule[0]
        timing = schedule[1]
        color = schedule[2]

        print('seq{}:{}'.format(i, sequence))
        print('time{}:{}'.format(i, timing))
        print('color{}:{}'.format(i, color))

    return 0


def random_rsvp_calibration_seq_gen(alp, timing=[0.5, 1, 0.2],
                                    color=['green', 'red', 'white'],
                                    num_sti=10,
                                    len_sti=10):
    """ Generates random RSVPKeyboard sequences.
        Args:
            alp(list[str]): alphabet (can be arbitrary)
            timing(list[float]): Task specific timing for generator
            color(list[str]): Task specific color for generator
                First element is the target, second element is the fixation
                Observe that [-1] element represents the trial information
            num_sti(int): number of random stimuli to be created
            len_sti(int): number of trials in a sequence
        Return:
            schedule_seq(tuple(
                samples[list[list[str]]]: list of sequences
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled sequences
            """

    len_alp = len(alp)

    samples, times, colors = [], [], []
    for idx_num in range(num_sti):
        rand_smp = np.random.choice(range(len_alp), len_sti)
        sample = [alp[rand_smp[0]], '+']
        rand_smp = np.random.permutation(rand_smp)
        sample += [alp[i] for i in rand_smp]
        samples.append(sample)
        times.append([timing[i] for i in range(len(timing) - 1)] +
                     [timing[-1]] * len_sti)
        colors.append([color[i] for i in range(len(color) - 1)] +
                      [color[-1]] * len_sti)

    schedule_seq = (samples, times, colors)

    return schedule_seq


def target_rsvp_sequence_generator(alp, target_letter, timing=[0.5, 1, 0.2],
                                   color=['green', 'white', 'white'],
                                   len_sti=10):

    """Generate target RSVPKeyboard sequences.

        Args:
            alp(list[str]): alphabet (can be arbitrary)
            target_letter([str]): letter to be copied
            timing(list[float]): Task specific timing for generator
            color(list[str]): Task specific color for generator
                First element is the target, second element is the fixation
                Observe that [-1] element represents the trial information
            len_sti(int): number of trials in a sequence
        Return:
            schedule_seq(tuple(
                samples[list[list[str]]]: list of sequences
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled sequences
    """

    len_alp = len(alp)

    # intialize our arrays
    samples, times, colors = [], [], []
    rand_smp = np.random.randint(0, len_alp, len_sti)
    sample = ['+']
    sample += [alp[i] for i in rand_smp]

    # if the target isn't in the array, replace it with some random index that
    #  is not fixation
    if target_letter not in sample:
        random_index = np.random.randint(0, len_sti - 1)
        sample[random_index + 1] = target_letter

    # add target letter to start
    sample = [target_letter] + sample

    # to-do shuffle the target letter

    samples.append(sample)
    times.append([timing[i] for i in range(len(timing) - 1)] +
                 [timing[-1]] * len_sti)
    colors.append([color[i] for i in range(len(color) - 1)] +
                  [color[-1]] * len_sti)

    schedule_seq = (samples, times, colors)

    return schedule_seq


def get_task_info(experiment_length, task_color):
    """ Generates fixed RSVPKeyboard task text and color information for
            display.
    Args:
        experiment_length(int): Number of sequences for the experiment
        task_color(str): Task information display color

    Return get_task_info((tuple): task_text: array of task text to display
                   task_color: array of colors for the task text
                   )
    """

    # Do list comprehensions to get the arrays for the task we need.

    task_text = ['%s/%s' % (stim + 1, experiment_length)
                 for stim in range(experiment_length)]
    task_color = [[str(task_color)] for stim in range(experiment_length)]

    return (task_text, task_color)


def _demo_random_rsvp_sequence_generator():
    alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']

    num_samples = int(np.random.randint(1, 10, 1))
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
        print('time{}:{}'.format(i, timing[i]))
        print('color{}:{}'.format(i, color[i]))
    return 0


def rsvp_copy_phrase_seq_generator(alp, target_letter, timing=[0.5, 1, 0.2],
                                   color=['green', 'white', 'white'],
                                   len_sti=10):
    """Generate copy phrase RSVPKeyboard sequences.

        Args:
            alp(list[str]): alphabet (can be arbitrary)
            target_letter([str]): letter to be copied
            timing(list[float]): Task specific timing for generator
            color(list[str]): Task specific color for generator
                First element is the target, second element is the fixation
                Observe that [-1] element represents the trial information
            len_sti(int): number of trials in a sequence
        Return:
            schedule_seq(tuple(
                samples[list[list[str]]]: list of sequences
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled sequences
    """

    len_alp = len(alp)

    # intialize our arrays
    samples, times, colors = [], [], []
    rand_smp = np.random.randint(0, len_alp, len_sti)
    sample = ['+']
    sample += [alp[i] for i in rand_smp]

    # if the target isn't in the array, replace it with some random index that
    #  is not fixation
    if target_letter not in sample:
        random_index = np.random.randint(0, len_sti - 1)
        sample[random_index + 1] = target_letter

    # to-do shuffle the target letter

    samples.append(sample)
    times.append([timing[i] for i in range(len(timing) - 1)] +
                 [timing[-1]] * len_sti)
    colors.append([color[i] for i in range(len(color) - 1)] +
                  [color[-1]] * len_sti)

    schedule_seq = (samples, times, colors)

    return schedule_seq


def main():
    # _demo_random_rsvp_sequence_generator()
    _demo_best_case_sequence_generator()

    return 0


if __name__ == "__main__":
    main()
