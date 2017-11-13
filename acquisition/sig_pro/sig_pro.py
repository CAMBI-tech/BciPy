import numpy as np
import os


def sig_pro(input_seq, filt=None, fs=256, k=2):
    """
    :param input_seq: Input sequence to be filtered. Expected dimensions are 16xT
    :param filt: Input for using a specific filter. If left empty, according to fs a pre-designed filter is going to be used. Filters are pre-designed for fs = 256,300 or 1024 Hz.
    :param fs: Sampling frequency of the hardware.
    :param k: downsampling order
    :return: output sequence that is filtered and downsampled input. Filter delay is compensated. Dimensions are 16xT/k

    256Hz
        - 1.75Hz to 45Hz
        - 60Hz -64dB Gain

    300Hz
        - 1.84Hz to 45Hz
        - 60Hz -84dB Gain

    1024Hz
        - 1.75Hz to 45Hz
        - 60Hz -64dB Gain

    """

    with open(os.path.dirname(os.path.abspath(__file__)) + '\\filters.txt',
              'r') as text_file:
        dict_of_filters = eval(text_file.readline())

    try:
        filt = dict_of_filters[fs]
    except Exception as e:
        print e
        print 'Please provide a filter for your sampling frequency.'

    filt = np.array(filt)
    filt = filt - np.sum(filt) / filt.size

    output_seq = [[]]

    # Convolution per channel
    for z in range(len(input_seq)):
        temp = np.convolve(input_seq[z][:], filt)
        # Filter off-set compensation
        temp = temp[int(np.ceil(len(filt) / 2.)) - 1:]
        # Downsampling
        output_seq.append(temp[::k])

    return np.array(output_seq[1:])
