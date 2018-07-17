import numpy as np
import random
import glob


def best_case_rsvp_seq_gen(alp, p, timing=[1, 0.2],
                           color=['red', 'white'], num_sti=1,
                           len_sti=10, is_txt=True):
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
        if is_txt:
            sample = ['+']
        else:
            sample = ['bcipy/static/images/bci_main_images/PLUS.png']
        idx = np.random.permutation(idx)
        sample += [alp[i] for i in idx]
        samples.append(sample)
        times.append([timing[i] for i in range(len(timing) - 1)] +
                     [timing[-1]] * len_sti)
        colors.append([color[i] for i in range(len(color) - 1)] +
                      [color[-1]] * len_sti)

    schedule_seq = (samples, times, colors)

    return schedule_seq


def random_rsvp_calibration_seq_gen(alp, timing=[0.5, 1, 0.2],
                                    color=['green', 'red', 'white'],
                                    num_sti=10,
                                    len_sti=10, is_txt=True):
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
        idx = np.random.permutation(np.array(list(range(len_alp))))
        rand_smp = (idx[0:len_sti])
        if not is_txt:
            sample = [
                alp[rand_smp[0]],
                'bcipy/static/images/bci_main_images/PLUS.png']
        else:
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


def target_rsvp_sequence_generator(alp, target_letter, parameters, timing=[0.5, 1, 0.2],
                                   color=['green', 'white', 'white'],
                                   len_sti=10, is_txt=True):

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
    rand_smp = random.sample(range(len_alp), len_sti)
    if is_txt:
        sample = ['+']
    else:
        sample = ['../bci/static/images/bci_main_images/PLUS.png']
        target_letter = parameters[
            'path_to_presentation_images'] + target_letter + '.png'
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

    # initialize our arrays
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

def generate_icon_match_images(experiment_length, image_path, number_of_sequences, timing):
    """Generates an array of images to use for the icon matching task.
    Args:
        experiment_length(int): Number of images per sequence
        image_path(str): Path to image files
        number_of_sequences(int): Number of sequences to generate
        timing(list): List of timings
    Return generate_icon_match_images(arrays of tuples of paths to images to
    display, and timings)
    """
    #Get all png images in image path
    image_array = glob.glob(image_path + '*.png')

    #Remove plus image from array
    for image in image_array:
        if image.endswith('PLUS.png'):
            image_array.remove(image)

    if experiment_length > len(image_array) - 1:
        raise Exception('Number of images to be displayed on screen is longer than number of images available')
        return

    #Generate indexes of target images
    target_image_numbers = np.random.randint(0, len(image_array), number_of_sequences)

    #Array of images to return
    return_array = []

    #Array of timings to return
    return_timing = []
    for specific_time in range(len(timing) - 1):
        return_timing.append(timing[specific_time])
    for item_without_timing in range(len(return_timing), experiment_length):
        return_timing.append(timing[-1])

    for sequence in range(number_of_sequences):
        return_array.append([])
        #Generate random permutation of image indexes
        random_number_array = np.random.permutation(len(image_array))
        #Add target image to image array
        return_array[sequence].append(image_array[target_image_numbers[sequence]])
        #Add PLUS.png to image array
        return_array[sequence].append('bcipy/static/images/bci_main_images/PLUS.png')

        #Add target image to sequence, if it is not already there
        if not target_image_numbers[sequence] in random_number_array[2:experiment_length]:
            random_number_array[np.random.randint(2, experiment_length)] = target_image_numbers[sequence]

        #Fill the rest of the image array with random images
        for item in range(2, experiment_length):
            return_array[sequence].append(image_array[random_number_array[item]])

    return (return_array, return_timing)
