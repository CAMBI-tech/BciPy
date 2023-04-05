import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

from bcipy.config import DEFAULT_PARAMETERS_PATH, TRIGGER_FILENAME, RAW_DATA_FILENAME
from bcipy.preferences import preferences
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_gaze_data
)
from bcipy.helpers.triggers import TriggerType, trigger_decoder

from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")

def subset_data(data: np.ndarray, labels: np.ndarray, test_size: float, random_state=0):
    """Performs a train/test split on the provided data and labels, accounting for
    the current shape convention (channel dimension in front, instead of batch dimension in front).

    Args:
        data (np.ndarray): Shape (channels, items, time)
        labels (np.ndarray): Shape (items,)
        test_size (float): fraction of data to be used for testing
        random_state (int, optional): fixed random seed

    Returns:
        train_data (np.ndarray): Shape (channels, train_items, time)
        test_data (np.ndarray): Shape (channels, test_items, time)
        train_labels (np.ndarray): Shape (train_items,)
        test_labels (np.ndarray): Shape (test_items,)
    """
    data = data.swapaxes(0, 1)
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )
    train_data = train_data.swapaxes(0, 1)
    test_data = test_data.swapaxes(0, 1)
    return train_data, test_data, train_labels, test_labels


def offline_gaze_analysis(
    data_folder: str = None,
    parameters: dict = {},
):
    
    if not data_folder:
        data_folder = load_experimental_data()

    # extract relevant session information from parameters file
    poststim_length = parameters.get("trial_length")
    prestim_length = parameters.get("prestim_length")
    trials_per_inquiry = parameters.get("stim_length")
    static_offset = parameters.get("static_trigger_offset")    
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)
    raw_data_file = f"{RAW_DATA_FILENAME}.csv"

    log.info(
        f"\nData processing settings: \n"
        f"Poststimulus: {poststim_length}s, Prestimulus: {prestim_length}s, Buffer: {buffer}s, Trials per inquiry: {trials_per_inquiry} \n"
    )
        
    # Load raw data
    gaze_data = load_raw_gaze_data(Path(data_folder, raw_data_file))
    channels = gaze_data.channels
    type_amp = gaze_data.daq_type
    sample_rate = gaze_data.sample_rate

    log.info(f"Channels read from csv: {channels}")
    log.info(f"Device type: {type_amp}, fs={sample_rate}")

    # Process triggers.txt files
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        remove_pre_fixation = False,
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )

    # Channel map can be checked from raw_data.csv file or the devices.json located in the acquisition module
    # The timestamp column [0] is already excluded.
    channel_map = analysis_channels(channels, type_amp)
    # channel_map = [0,0,1,1,0,1,1,0,0] 

    data, fs = gaze_data.by_channel()  # no transformations applied at this stage

    breakpoint()
    # reshaper!
    results = []

    #TODO: 
    # Implement reshaper. We might not need a separate reshaper function.
    # Plotting for offset
    # Plot the gaze data (check NSLR) fixation, saccade, blink
    # Coordinates in Tobii for targets in each inquiry (Tobii to Psychopy units)


    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", default=None)
    parser.add_argument("-p", "--parameters_file", default=DEFAULT_PARAMETERS_PATH)
    args = parser.parse_args()

    log.info(f"Loading params from {args.parameters_file}")
    parameters = load_json_parameters(args.parameters_file, value_cast=True)

    offline_gaze_analysis(
        args.data_folder,
        parameters)
    log.info("Offline Analysis complete.")