
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
)
from bcipy.helpers.process import load_data_inquiries, load_data_mne
from bcipy.helpers.visualization import visualize_erp, visualize_evokeds, visualize_joint_average
from bcipy.config import DEFAULT_PARAMETERS_PATH

import mne
from mne import grand_average
from mne.viz import plot_compare_evokeds


from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")

# ARTIFACT_LABELLED_FILENAME = 'artifacts_raw.fif'
ARTIFACT_LABELLED_FILENAME = 'auto_artifacts_raw.fif'

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
    plot_joint_times = [-0.1, 0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]

    parameters = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True)

    non_target = []
    target = []
    all_epochs = []
    for session in Path(data_path).iterdir():
        if session.is_dir():

            try:
                session_folder = str(session.resolve())
                # raw_data, data, labels, trigger_timing, channel_map, poststim_length, def_transform, _ = load_data_inquiries(
                #                     data_folder=session_folder)
                # epochs, figs = visualize_erp(raw_data, channel_map, trigger_timing, labels, 0.8, def_transform, show=False) # TODO create my own epochs to avoid all the figs

                mne_data = mne.io.read_raw_fif(f'{session}/{ARTIFACT_LABELLED_FILENAME}')
                raw_data, trial_data, labels, trigger_timing, channel_map, poststim_length, default_transform, dl, epochs  = load_data_mne(
                    data_folder=session_folder,
                    mne_data_annotations=mne_data.annotations,
                    drop_artifacts=True,
                    trial_length=0.65,
                    parameters=parameters)

                # average the epochs
                non_target.append(epochs['1'])
                target.append(epochs['2'])
                all_epochs.append(epochs)
            except Exception as e:
                print(f"Error processing session {session}: \n {e}")
                breakpoint()

    
    # concatenate the epochs
    all_mne_epochs = mne.concatenate_epochs(all_epochs)
    target = mne.concatenate_epochs(target)
    target_evoked = target.average()
    non_target = mne.concatenate_epochs(non_target)
    non_target_evoked = non_target.average()
    all_data_epochs = mne.concatenate_epochs([non_target, target])
    
    data = [non_target, target]
    visualize_evokeds(data, show=True)
        
    # plot the averages using grand_average
    # target_evoked, non_target_evoked = group_average(target, non_target)
    # breakpoint()
    target_evoked.plot_image()
    non_target_evoked.plot_image()
    visualize_joint_average((non_target, target), ['Non-Target', 'Target'], show=True, plot_joint_times=plot_joint_times)

    breakpoint()
    # Show a ERP of the two events using gfp by default and no CI?
    plot_compare_evokeds([target_evoked, non_target_evoked], combine='mean', show=True )
    # plot_compare_evokeds([target.iter_evoked(), non_target.iter_evoked()], combine='mean', show=True)
    plot_compare_evokeds(all_mne_epochs, combine='mean')

    # Shows a heatmaps of activation in mv by channel
    # target_evoked.plot_image()
    # non_target_evoked.plot_image()

    # plot the joint average
    # visualize_joint_average((non_target, target), ['Non-Target', 'Target'], show=True, plot_joint_times=plot_joint_times)
    # visualize_joint_average(all_mne_epochs, ['Non-Target', 'Target'], show=True, plot_joint_times=plot_joint_times)