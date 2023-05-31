from bcipy.config import (TRIGGER_FILENAME,
                          RAW_DATA_FILENAME,
                          DEFAULT_DEVICE_SPEC_FILENAME)
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.visualization import visualize_erp, visualize_evokeds, visualize_joint_average
from bcipy.config import DEFAULT_PARAMETER_FILENAME
from bcipy.helpers.triggers import TriggerType, trigger_decoder

import bcipy.acquisition.devices as devices
from bcipy.signal.process.transform import dummy_transform
from bcipy.signal.process import filter_inquiries, get_default_transform
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.helpers.stimuli import update_inquiry_timing

from mne import grand_average


from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")

def group_average(target_evoked, nontarget_evoked):
    """Given a list of target and nontarget evokeds, return a grand average of both."""
    target_evoked = grand_average(target_evoked)
    nontarget_evoked = grand_average(nontarget_evoked)
    return target_evoked, nontarget_evoked

if __name__ == "__main__":
    """Process all sessions in a data folder and print the AUCs.
    
    This script is intended to be run from the command line with the following command:
        python offline_analysis_process_dataset.py
    """
    path = load_experimental_data()
    plot_joint_times = [-0.1, 0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]

    non_target = []
    target = []
    for session in Path(path).iterdir():
        if session.is_dir():

            try:
                data, labels, trigger_timing, channel_map, poststim_length = load_data(
                                    data_folder=str(session.resolve()))
            
                epochs, figs = visualize_erp(data, labels, trigger_timing, channel_map, poststim_length)
                # average the epochs
                non_target.append(epochs['1'].average())
                target.append(epochs['2'].average())
            except Exception as e:
                print(f"Error processing session {session}: \n {e}")
                breakpoint()
        
    # plot the averages using grand_average
    target, non_target = group_average(target, non_target)

    # visualize_evokeds((non_target, target), show=True)

    # plot the joint average
    # visualize_joint_average((non_target, target), ['Non-Target', 'Target'], show=True, times=plot_joint_times)