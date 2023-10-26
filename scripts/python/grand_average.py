
from bcipy.helpers.load import (
    load_experimental_data,
)
from bcipy.helpers.process import load_data_inquiries, load_data_mne
from bcipy.helpers.visualization import visualize_erp, visualize_evokeds, visualize_joint_average

import mne
from mne import grand_average
from mne.viz import plot_compare_evokeds


from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")

ARTIFACT_LABELLED_FILENAME = 'artifacts_raw.fif'

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
                session_folder = str(session.resolve())
                # raw_data, data, labels, trigger_timing, channel_map, poststim_length, def_transform, _ = load_data_inquiries(
                #                     data_folder=session_folder)
                # epochs, figs = visualize_erp(raw_data, channel_map, trigger_timing, labels, 0.8, def_transform, show=False) # TODO create my own epochs to avoid all the figs

                mne_data = mne.io.read_raw_fif(f'{session}/{ARTIFACT_LABELLED_FILENAME}')
                raw_data, trial_data, labels, trigger_timing, channel_map, poststim_length, default_transform, dl, epochs  = load_data_mne(
                    data_folder=session_folder, mne_data_annotations=mne_data.annotations, drop_artifacts=False, trial_length=0.8)

                # average the epochs
                non_target.append(epochs['1'])
                target.append(epochs['2'])
            except Exception as e:
                print(f"Error processing session {session}: \n {e}")
                breakpoint()

    
    # concatenate the epochs
    target = mne.concatenate_epochs(target)
    target_evoked = target.average()
    non_target = mne.concatenate_epochs(non_target)
    non_target_evoked = non_target.average()
    all_data_epochs = mne.concatenate_epochs([non_target, target])

    # visualize_evokeds(all_data_epochs, show=True)
        
    # plot the averages using grand_average
    # target_evoked, non_target_evoked = group_average(target, non_target)
    breakpoint()

    # Show a ERP of the two events using gfp by default and no CI?
    plot_compare_evokeds([target, non_target], combine='mean')

    # Shows a heatmaps of activation in mv by channel
    # target_evoked.plot_image()
    # non_target_evoked.plot_image()

    # plot the joint average
    # visualize_joint_average((non_target, target), ['Non-Target', 'Target'], show=True, plot_joint_times=plot_joint_times)