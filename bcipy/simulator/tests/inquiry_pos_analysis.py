""" investigating inquiry position on model responses """
import numpy as np

from bcipy.helpers import session
from bcipy.task.data import EvidenceType
from matplotlib import pyplot as plt

""" 
Approach:

Read in an experiment path, from main function
Parse the session.json object and group likelihood values based on stimuli position

"""


# plotting target or nontarget
def plot_groups(group, title="", clip=None, color='blue'):
    data = list(group.values())
    # Create a figure and a 2x5 grid of subplots (2 rows, 5 columns)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))

    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()

    # Iterate over the data and the flattened axes together using 'zip'
    count = 0
    for ax, data_to_plot in zip(axes_flat, data):
        clipped_vals = np.clip(data_to_plot, clip[0], clip[1]) if clip is not None else None
        ax.hist(clipped_vals if clipped_vals is not None else data_to_plot, bins=15, color=color,
                alpha=0.7)
        ax.set_title(f'Position {count}')
        ax.set_xlabel('Value')
        count += 1

    plt.savefig(f"bcipy/simulator/tests/resource/{title}.png")
    plt.clf()


if __name__ == "__main__":
    experiment_path = "/Users/srikarananthoju/cambi/tab_test_dynamic/16sec_-0700"
    # session_path = f"{experiment_path}/session.json"

    wrapper_path = f"/Users/srikarananthoju/cambi/tab_test_dynamic"
    session_paths = [f"{wrapper_path}/16sec_-0700/session.json",
                     f"{wrapper_path}/50sec_-0700/session.json",
                     f"{wrapper_path}/29sec_-0700/session.json"]

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

    plot_groups(groups_nontarget, clip=(0, 1), title="nontarget")
    plot_groups(groups_target, color='red', title="target")
