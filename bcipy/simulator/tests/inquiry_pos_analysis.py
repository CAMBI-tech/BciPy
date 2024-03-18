""" investigating inquiry position on model responses """
from typing import List

import numpy as np

from bcipy.helpers import session
from bcipy.task.data import EvidenceType
from matplotlib import pyplot as plt

""" 
Approach:

Read in an experiment path, from main function
Parse the session.json object and group likelihood values based on stimuli position

"""


def round_to_nearest(num, k):
    """ Rounds number to nearest k. Ex: f(66, 25) => 75 """
    return round(num / k) * k


def normalize(vec: List):
    """ sample normalizaiton of list """

    summation = sum(vec)
    new_vec = [val / summation for val in vec]
    return new_vec


# plotting target or nontarget
def plot_groups(group, title="", clip=None, color='blue'):
    data = list(group.values())
    sublist_counts = [len(sublist) for sublist in data]
    normalized_data = map(normalize, data)
    # Create a figure and a 2x5 grid of subplots (2 rows, 5 columns)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))

    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()

    max_y_val = round_to_nearest(max(sublist_counts), 20)

    # Iterate over the data and the flattened axes together using 'zip'
    count = 0
    for ax, data_to_plot in zip(axes_flat, normalized_data):
        # clipping max bin for histogram
        clipped_vals = np.clip(data_to_plot, clip[0], clip[1]) if clip is not None else None
        bin_edges = np.arange(0, 1.1, 0.05)  # setting constant bins
        ax.hist(clipped_vals if clipped_vals is not None else data_to_plot, bins=bin_edges,
                color=color,
                alpha=0.7)

        ax.set_yticks([0, max_y_val])  # cleaning y axis

        ax.set_title(f'Position {count}')
        ax.set_xlabel('EEG')
        count += 1

    fig.suptitle(title)
    plt.savefig(f"bcipy/simulator/tests/resource/{title}_hist.png")
    plt.clf()


if __name__ == "__main__":
    # edit these for different data sources and add to session_tuples
    wrapper_path1 = f"/Users/srikarananthoju/cambi/Dan_matrix"
    session_folders1 = ["Dan_Matrix_Copy_Phrase_Wed_10_Jan_2024_12hr32min08sec_-0800",
                        "Dan_Matrix_Copy_Phrase_Wed_10_Jan_2024_12hr36min06sec_-0800",
                        "Dan_Matrix_Copy_Phrase_Wed_10_Jan_2024_12hr45min46sec_-0800",
                        "Dan_Matrix_Copy_Phrase_Wed_10_Jan_2024_12hr53min23sec_-0800",
                        "Dan_Matrix_Copy_Phrase_Wed_10_Jan_2024_12hr54min10sec_-0800"]

    wrapper_path2 = "/Users/srikarananthoju/cambi/tab_test_dynamic"
    session_folders2 = [f"16sec_-0700",
                        f"50sec_-0700",
                        f"29sec_-0700"]

    session_tuples = []
    session_tuples.append((wrapper_path1, session_folders1))
    session_tuples.append((wrapper_path2, session_folders2))

    session_paths = [f"{tup[0]}/{sf}/session.json" for tup in session_tuples for sf in tup[1]]

    groups_target = {i: [] for i in range(10)}  # inq position to eeg evidences for targets
    groups_nontarget = {i: [] for i in range(10)}

    for session_path in session_paths:
        loaded_session = session.read_session(file_name=session_path)

        stim_alp_idx = {v: i for i, v in
                        enumerate(loaded_session.symbol_set)}  # alphabet symbol => idx

        for series in loaded_session.series:
            for inq in series:
                stimuli = inq.stimuli[1:]  # removing fixation cross
                target_info = inq.target_info[1:]

                for position, symbol in enumerate(stimuli):
                    eeg_model_response = inq.eeg_evidence[stim_alp_idx[symbol]]  # lik for symbol
                    if target_info[position] == 'target':
                        groups_target[position].append(eeg_model_response)
                    else:
                        groups_nontarget[position].append(eeg_model_response)

    title = "dan+tab"  # change plot title
    plot_groups(groups_nontarget, clip=(0, 1), title=f"{title} nontarget")
    plot_groups(groups_target, color='red', title=f"{title} target")
