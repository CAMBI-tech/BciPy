""" timing and color information either hard coded or read from a parameters
file """
import numpy as np


def random_clb_rsvp_seq_gen(alp, timing=[0.5, 1, 0.2],
                            color=['green', 'red', 'white'], num_sti=10,
                            len_sti=10):
    """ Generates random RSVPKeyboard sequences
        Args:
            alp(list[str]): alphabet (can be arbitrary)
            timing(list[float]): Task specific timing for generator
            color(list[str]): Task specific color for generator
                First element is the target, second element is the fixation
                Observe that [-1] element represents the trial information
            num_sti(int): number of random stimuli to be created
            len_sti(int): number of trials in a sequence
        Return:
            schedule_seq(dict{
                samples[list[list[str]]]: list of sequences
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors}): scheduled sequences
            """

    len_alp = len(alp)

    samples, times, colors = [], [], []
    for idx_num in range(num_sti):
        rand_smp = np.random.randint(0, len_alp, len_sti)
        sample = [alp[rand_smp[0]], '+']
        sample += [alp[i] for i in rand_smp]
        samples.append(sample)
        times.append([timing[i] for i in range(len(timing) - 1)] +
                     [timing[-1]] * len_sti)
        colors.append([color[i] for i in range(len(color) - 1)] +
                      [color[-1]] * len_sti)

    schedule_seq = (samples, times, colors)

    return schedule_seq


def _demo_random_clb_rsvp_seq_gen():
    alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']

    num_samples = int(np.random.randint(1, 10, 1))
    len_samples = int(np.random.randint(1, 20, 1))

    print('Number of sequences:{}, Number of trials:{}'.format(num_samples,
                                                               len_samples))

    print('Alphabet:{}'.format(alp))
    schedule = random_clb_rsvp_seq_gen(alp=alp, num_sti=num_samples,
                                       len_sti=len_samples)
    sequences = schedule[0]
    timing = schedule[1]
    color = schedule[2]

    for i in range(len(sequences)):
        print('seq{}:{}'.format(i, sequences[i]))
        print('time{}:{}'.format(i, timing[i]))
        print('color{}:{}'.format(i, color[i]))
    return 0


def main():
    _demo_random_clb_rsvp_seq_gen()

    return 0


if __name__ == "__main__":
    main()
