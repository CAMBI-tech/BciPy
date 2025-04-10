"""Demo script to load BIDS data into MNE format using BciPy.

This script demonstrates how to load BIDS data using the BciPy library. This assumes that the data is
already in BIDS format (BV, EDF, or BDF, with the channels.tsv file) and that the BIDS directory is
structured correctly. 

The BIDS directory should contain a dataset_description.json file and a
participants.tsv file, along with the data files in the appropriate subdirectories.
"""

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
    raw_data_files = BIDS_to_MNE(path_to_bids, verbose=True)

    breakpoint()  # Set a breakpoint here to inspect the raw_data variable in your debugger.

    raw_data = raw_data_files[0]  # Get the first raw data object from the list.
    # PLOT THE RAW DATA
    # take the raw data object and plot the data
    # raw_data.plot()  # Uncomment this line to plot the raw data.
    # Note: The plot function is commented out to avoid opening a plot window during the demo.

    # EPOCH THE DATA / CREATE ERPS
    # epoch the data using the events from the raw data object.
    # This will create a new raw data object with the epoched data.
    # You can specify the event_id, tmin, tmax, and baseline parameters as needed.
    # events = mne.find_events(raw_data)  # Find events in the raw data.
    # event_id = {'event_name': 1}  # Specify the event ID for the epochs.
    # tmin = -0.2  # Start time before the event.
    # tmax = 0.5  # End time after the event.
    # baseline = (None, None)  # No baseline correction.
    # epochs = mne.Epochs(raw_data, events, event_id, tmin, tmax, baseline, preload=True)
    # epochs.plot()  # Uncomment this line to plot the epoched data.

except Exception as e:
    print(f"An error occurred: {e}")
