
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

import matplotlib.pyplot as plt


from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")

ARTIFACT_LABELLED_FILENAME = 'artifacts_raw.fif'
AUTO_ARTIFACT_LABELLED_FILENAME = 'auto_artifacts_raw.fif'

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
    plot_joint_times = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]

    parameters = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True)

    non_target_cf = []
    target_cf = []
    all_epochs_cf = []
    non_target_of = []
    target_of = []
    all_epochs_of = []
    for session in Path(data_path).iterdir():
        if session.is_dir():

            try:
                session_folder = str(session.resolve())

                # Online Filtering Data
                raw_data, data, labels, trigger_timing, channel_map, poststim_length, def_transform, _, channels = load_data_inquiries(
                                    data_folder=session_folder,
                                    trial_length=0.76,
                                    apply_filter=True)
                
                # turn data and labels into epochs
                channel_types = ['eeg'] * len(channels)
                mne_info = mne.create_info(channels, sfreq=150, ch_types='eeg')
                ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
                epochs = mne.EpochsArray(data, info=mne_info)
                epochs.set_montage(ten_twenty_montage)
                # epochs.apply_function(lambda x: x * 1e-6)
                nt = []
                t = []
                for i, label in enumerate(labels):
                    if label == 0:
                        nt.append(epochs[i])
                    else:
                        t.append(epochs[i])
                non_target_of.append(mne.concatenate_epochs(nt))
                target_of.append(mne.concatenate_epochs(t))
                all_epochs_of.append(epochs)

                # breakpoint()
                # CF Data
                # mne_data = mne.io.read_raw_fif(f'{session}/{ARTIFACT_LABELLED_FILENAME}')
                raw_data, trial_data, labels, trigger_timing, channel_map, poststim_length, default_transform, dl, epochs  = load_data_mne(
                    data_folder=session_folder,
                    # mne_data_annotations=mne_data.annotations,
                    drop_artifacts=False,
                    trial_length=0.76,
                    parameters=parameters)

                non_target_cf.append(epochs['1'])
                target_cf.append(epochs['2'])
                all_epochs_cf.append(epochs)

            except Exception as e:
                print(f"Error processing session {session}: \n {e}")
                breakpoint()

    

    # all_data_epochs = mne.concatenate_epochs([non_target_of, target_of])
    # REGULAR CF
    trial_length = 0.65
    target_cf = mne.concatenate_epochs(target_cf)
    target_cf = target_cf.crop(tmin=0.0, tmax=trial_length)
    target_cf_evoked = target_cf.average()
    non_target_cf = mne.concatenate_epochs(non_target_cf)
    non_target_cf = non_target_cf.crop(tmin=0.0, tmax=trial_length)
    non_target_cf_evoked = non_target_cf.average()

    # OF
    target_of = mne.concatenate_epochs(target_of)
    target_of = target_of.crop(tmin=0.0, tmax=trial_length)
    target_of_evoked = target_of.average()
    non_target_of = mne.concatenate_epochs(non_target_of)
    non_target_of = non_target_of.crop(tmin=0.0, tmax=trial_length)
    non_target_of_evoked = non_target_of.average()
    
    # data = [non_target_of, target_of]
    # visualize_evokeds(data, show=True)

    evokeds = { 'Online Filter Target': target_of_evoked, 'Online Filter Non-Target': non_target_of_evoked, 'Conventional Filter Non-Target': non_target_cf_evoked, 'Conventional Filter Target': target_cf_evoked }
    
    # Figure 1.
    roi = ['Pz', 'Cz', 'P3', 'P4', 'O1', 'O2']

    evokeds_erp_compare = dict(
        OF_target=list(target_of.iter_evoked()), 
        CF_target=list(target_cf.iter_evoked()),
        OF_non_target=list(non_target_of.iter_evoked()),
        CF_non_target=list(non_target_cf.iter_evoked())
    )

    evokeds_erp_compare_targets = dict(
        OF_target=list(target_of.iter_evoked()), 
        CF_target=list(target_cf.iter_evoked()),
    )

    evokeds_erp_compare_non_targets = dict(
        OF_non_target=list(non_target_of.iter_evoked()),
        CF_non_target=list(non_target_cf.iter_evoked())
    )

    
    breakpoint()
    fig = mne.viz.plot_compare_evokeds(evokeds_erp_compare, combine='mean', show=True, picks=roi, ci=0.95)
    fig_target_gfp = mne.viz.plot_compare_evokeds(evokeds_erp_compare_targets, combine='gfp', show=True)
    fig_non_target_gfp = mne.viz.plot_compare_evokeds(evokeds_erp_compare_non_targets, combine='gfp', show=True)
    # Show a ERP of the two events using gfp by default and no CI?
    # visualize_evokeds([target_cf, target_of], show=True)
    target_compare = dict(
        CF_target=target_cf.average(),
        OF_target=target_of.average()
    )
    all_compare = dict(
        CF_target=target_cf.average(),
        OF_target=target_of.average(),
        CF_non_target=non_target_cf.average(),
        OF_non_target=non_target_of.average()
    )
    plot_compare_evokeds(all_compare, title='ERP Difference Between Online and Conventional Filtering', combine='gfp', show=True)
    # eco = plot_compare_evokeds([cf_evokeds_diff, of_evokeds_diff], picks=roi, title='ERP Difference Between Online and Conventional Filtering')
    # plot_compare_evokeds(evokeds, picks=roi, title='ERP Difference Between Online and Conventional Filtering')
    # plt.show()
    # Shows a heatmaps of activation in mv by channel
    # target_evoked.plot_image()
    # non_target_evoked.plot_image()
    breakpoint()
