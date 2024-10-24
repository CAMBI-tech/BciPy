from bcipy.helpers.stimuli import StimuliOrder, generate_calibration_inquiries, best_case_rsvp_inq_gen
import numpy as np


def _demo_random_rsvp_inquiry_generator():
    alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']

    num_samples = 2
    len_samples = 10

    print('Number of inquiries:{}, Number of trials:{}'.format(num_samples,
                                                               len_samples))

    print('Alphabet:{}'.format(alp))
    schedule = generate_calibration_inquiries(alp=alp,
                                              inquiry_count=num_samples,
                                              stim_per_inquiry=len_samples,
                                              stim_order=StimuliOrder.RANDOM)
    inquiries = schedule[0]
    timing = schedule[1]
    color = schedule[2]

    for i in range(len(inquiries)):
        print('inq{}:{}'.format(i, inquiries[i]))
        if len(set(inquiries[i][2::])) != len(inquiries[i][2::]):
            raise Exception('Letter repetition!')
        print('time{}:{}'.format(i, timing[i]))
        print('color{}:{}'.format(i, color[i]))


def _demo_alphabetical_rsvp_inquiry_generator():
    alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']

    num_samples = 2
    len_samples = 10

    print('Number of inquiries:{}, Number of trials:{}'.format(num_samples,
                                                               len_samples))

    print('Alphabet:{}'.format(alp))
    schedule = generate_calibration_inquiries(alp=alp,
                                              inquiry_count=num_samples,
                                              stim_per_inquiry=len_samples,
                                              stim_order=StimuliOrder.ALPHABETICAL)
    inquiries = schedule[0]
    timing = schedule[1]
    color = schedule[2]

    for i in range(len(inquiries)):
        print('inq{}:{}'.format(i, inquiries[i]))
        if len(set(inquiries[i][2::])) != len(inquiries[i][2::]):
            raise Exception('Letter repetition!')
        print('time{}:{}'.format(i, timing[i]))
        print('color{}:{}'.format(i, color[i]))


def _demo_best_case_inquiry_generator():
    alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']

    num_samples = 2
    len_samples = 10

    print('Number of inquiries:{}, Number of trials:{}'.format(num_samples,
                                                               len_samples))
    print('Alphabet:{}'.format(alp))
    for i in range(num_samples):
        p = np.random.randint(0, 10, len(alp))
        print('Probabilities:{}'.format(p))
        schedule = best_case_rsvp_inq_gen(alp=alp, session_stimuli=p, stim_number=1,
                                          stim_length=len_samples)
        inquiry = schedule[0]
        timing = schedule[1]
        color = schedule[2]

        print('inq{}:{}'.format(i, inquiry))
        print('time{}:{}'.format(i, timing))
        print('color{}:{}'.format(i, color))

    return 0


if __name__ == '__main__':
    print('\n\n\n====== Best Case =======\n')
    _demo_best_case_inquiry_generator()

    print('\n\n\n====== Random RSVP Inquiry Generator =======\n')
    _demo_random_rsvp_inquiry_generator()

    print('\n\n\n====== Alphabetical RSVP Inquiry Generator =======\n')
    _demo_alphabetical_rsvp_inquiry_generator()
