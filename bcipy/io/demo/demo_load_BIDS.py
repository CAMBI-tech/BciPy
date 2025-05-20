"""Demo script to load BIDS data into MNE format using BciPy.

This script demonstrates how to load BIDS data using the BciPy library. This assumes that the data is
already in BIDS format (BV, EDF, or BDF, with the channels.tsv file) and that the BIDS directory is
structured correctly.

The BIDS directory should contain a dataset_description.json file and a
participants.tsv file, along with the data files in the appropriate subdirectories.
"""

import mne
from mne.viz import plot_compare_evokeds

from bcipy.io.convert import BIDS_to_MNE
from bcipy.io.load import load_experimental_data

# This script demonstrates how to load BIDS data using the BciPy library. This assumes that the data is
# already in BIDS format (BV, EDF, or BDF, with the channels.tsv file) and that the BIDS directory is
# structured correctly. The BIDS directory should contain a dataset_description.json file and a
# participants.tsv file, along with the data files in the appropriate subdirectories.
try:
    path_to_bids = load_experimental_data()

    # Load BIDS data using MNE-Python and MNE-BIDS. This returns a list of raw data objects.
    # The data is loaded from the BIDS directory specified in the path_to_bids variable.
    # task_name = 'Calibration'  # Specify the task name to load data for.
    # This will filter the data to only include the specified task.
    task_name = None  # Set to None to load all tasks.
    raw_data_files = BIDS_to_MNE(path_to_bids, task_name=task_name)

    raw_data = raw_data_files[0]  # Get the first raw data object from the list.

    # to see where the data is stored, you can use the following command:
    # print(raw_data.filenames)

    # PLOT THE RAW DATA
    raw_data.load_data()
    raw_data.filter(1, 20)  # Apply a bandpass filter to the data (1-20 Hz).
    raw_data.plot()  #

    # EPOCH THE DATA / CREATE ERPS
    # epoch the data using the events from the raw data object.
    # You can specify the event_id, tmin, tmax, and baseline parameters as needed.
    events = mne.events_from_annotations(raw_data, event_id={'nontarget': 0, 'target': 1})
    tmin = -0.2
    tmax = 0.8
    baseline = (None, None)  # No baseline correction.
    epochs = mne.Epochs(raw_data, events[0], events[1], tmin, tmax, baseline=baseline, preload=True)

    # Grab the epochs for non-target and target events.
    non_target_epochs = epochs['nontarget']
    target_epochs = epochs['target']

    # Create evoked objects for non-target and target events.
    non_target_evoked = non_target_epochs.average()
    target_evoked = target_epochs.average()

    # PLOT THE EVOKED DATA
    plot_compare_evokeds(
        dict(
            nontarget=non_target_evoked,
            target=target_evoked),
        title='Evoked Responses for Non-Target and Target Events',
        show_sensors=True,
        ci=0.95,
        combine='mean',
        invert_y=True)
    target_evoked.plot_joint()
    non_target_evoked.plot_joint()

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Demo script completed.")
    breakpoint()  # This will pause the script execution and allow you to inspect the variables in the debugger.
