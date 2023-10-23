
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.process import load_data_inquiries, load_data_mne
from bcipy.helpers.visualization import visualize_erp, visualize_evokeds, visualize_joint_average

from mne import grand_average
from mne.viz import plot_compare_evokeds


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
    data_path = load_experimental_data()
    plot_joint_times = [-0.1, 0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]

    non_target = []
    target = []
    for session in Path(data_path).iterdir():
        if session.is_dir():

            try:
                # TODO visualize without artifact handling
                raw_data, data, labels, trigger_timing, channel_map, poststim_length, def_transform, _ = load_data_inquiries(
                                    data_folder=str(session.resolve()))
            
                epochs, figs = visualize_erp(raw_data, channel_map, trigger_timing, labels, 0.8, def_transform, show=False)
                # average the epochs
                non_target.extend(list(epochs[0].iter_evoked()))
                target.append(list(epochs[1].iter_evoked()))
            except Exception as e:
                print(f"Error processing session {session}: \n {e}")
                breakpoint()
        
    # plot the averages using grand_average
    target_evoked, non_target_evoked = group_average(target, non_target)
    breakpoint()

    # Show a ERP of the two events using gfp by default and no CI?
    # plot_compare_evokeds([target, non_target])

    # Shows a heatmaps of activation in mv by channel
    # target_evoked.plot_image()
    # non_target_evoked.plot_image()

    # plot the joint average
    # visualize_joint_average((non_target, target), ['Non-Target', 'Target'], show=True, times=plot_joint_times)