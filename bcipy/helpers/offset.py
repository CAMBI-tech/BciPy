from typing import List
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import normaltest, describe

from bcipy.helpers.triggers import RawData, clock_seconds
from bcipy.config import TRIGGER_FILENAME

def calculate_latency(raw_data: RawData,
                      diode_channel: str,
                      triggers: List[tuple],
                      plot_title: str = "",
                      recommend_static: bool = False,
                      plot: bool = False,
                      tolerance: float = 0.01):
    """Plot raw_data triggers, including the TRG_device_stream data
    (channel streamed from the device; usually populated from a trigger box),
    as well as TRG data populated from the LSL Marker Stream. Also plots data
    from the triggers.txt file if available.

    Parameters:
    -----------
        raw_data - complete list of samples read in from the raw_data.csv file.
        triggers - list of (trg, trg_type, stamp) values from the triggers.txt
            file. Stamps have been converted to the acquisition clock using
            the offset recorded in the triggers.txt.
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel('acquisition clock (secs)')
    plt.ylabel('TRG value')
    if plot_title:
        plt.title("\n".join(wrap(plot_title, 50)))

    # Plot TRG column; this is a continuous line with >0 indicating light.
    trg_box_channel = raw_data.dataframe[diode_channel]
    trg_box_y = []
    trg_ymax = 1.5
    trg_box_x = []

    # setup some analysis variable
    diode_enc = False
    starts = []
    lengths = []
    length = 0

    # Get the photodiode trigger data in the form of a list of timestamps
    for i, val in enumerate(trg_box_channel):
        timestamp = clock_seconds(raw_data.sample_rate, i + 1)
        value = int(float(val))
        trg_box_x.append(timestamp)
        trg_box_y.append(value)

        if value > 0 and not diode_enc:
            diode_enc = True
            starts.append(timestamp)

        if value > 0 and diode_enc:
            length += 0

        if value < 1 and diode_enc:
            diode_enc = False
            lengths.append(length)
            length = 0

    ax.plot(trg_box_x, trg_box_y, label=f'{diode_channel} (photodiode triggers)')

    # Plot triggers.txt data if present; vertical line for each value.
    if triggers:
        trigger_diodes_timestamps = [
            stamp for (_name, _trgtype, stamp) in triggers if _name == '\u25A0'
        ]
        plt.vlines(trigger_diodes_timestamps,
                   ymin=-1.0,
                   ymax=trg_ymax,
                   label=f'{TRIGGER_FILENAME} (adjusted)',
                   linewidth=0.5,
                   color='cyan')

    errors = []
    diffs = []
    x = 0

    # In the case of ending the session before the triggers are finished or the photo diode is detected at the end, 
    # we balance the timestamps
    if (len(starts) != len(trigger_diodes_timestamps)):
        if len(starts) > len(trigger_diodes_timestamps):
            starts = starts[:len(trigger_diodes_timestamps)]
        else:
            trigger_diodes_timestamps = trigger_diodes_timestamps[:len(starts)]

    for trigger_stamp, diode_stamp in zip(trigger_diodes_timestamps, starts):
        diff = trigger_stamp - diode_stamp

        # TODO: RSVP and Matrix Calibration Task: Correct for fixation causing a false positive
        if x > 0:
            if abs(diff) > tolerance:
                errors.append(
                    f'trigger={trigger_stamp} diode={diode_stamp} diff={diff}'
                )
            diffs.append(diff)
        if x == 4:
            x = 0
        else:
            x += 1

    if recommend_static:
        # test for normality
        _, p_value = normaltest(diffs)
        print(f'{describe(diffs)}')

        # if it's not normal, take the median
        if p_value < 0.05:
            print('Non-normal distribution')
        recommended_static = abs(np.median(diffs))
        print(
            f'System recommended static offset median=[{recommended_static}]')
        recommended_static = abs(np.mean(diffs))
        print(f'System recommended static offset mean=[{recommended_static}]')

    else:
        if errors:
            num_errors = len(errors)
            print(
                f'RESULTS: Allowable tolerance between triggers and photodiode exceeded. {errors}'
                f'Number of violations: {num_errors}')
        else:
            print(
                f'RESULTS: Triggers and photodiode timestamps within limit of [{tolerance}s]!'
            )

    # display the plot
    if plot:
         # Add labels for TRGs
        first_trg = trigger_diodes_timestamps[0]

        # Set initial zoom to +-5 seconds around the calibration_trigger
        if first_trg:
            ax.set_xlim(left=first_trg - 5, right=first_trg + 5)

        ax.grid(axis='x', linestyle='--', color="0.5", linewidth=0.4)
        plt.legend(loc='lower left', fontsize='small')
        plt.show()

        # plot a scatter of the differences
        plt.scatter(range(len(diffs)), diffs)
        plt.show()