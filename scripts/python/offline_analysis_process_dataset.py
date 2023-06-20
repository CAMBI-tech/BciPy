from bcipy.helpers.visualization import visualize_erp, visualize_evokeds, visualize_joint_average
from bcipy.helpers.process import load_data_inquiries, load_data_trials, load_data_mne
from bcipy.helpers.load import load_experimental_data, load_json_parameters
from soa_classifier import crossvalidate_record, scores
from bcipy.config import DEFAULT_PARAMETER_FILENAME

from mne import grand_average

def group_average(target_evoked, nontarget_evoked):
    """Given a list of target and nontarget evokeds, return a grand average of both."""
    target_evoked = grand_average(target_evoked)
    nontarget_evoked = grand_average(nontarget_evoked)
    return target_evoked, nontarget_evoked

from pathlib import Path
import mne
import pandas as pd
import numpy as np
import csv


ARTIFACT_LABELLED_FILENAME = 'artifacts_raw.fif'

"""
Processing notes:

- OF: The data is loaded, reshaped to inquiries, filtered, then reshaped to trials. load_data_inquiries() is used.
- CF: The data is loaded, filtered, then reshaped to trials. load_data_trials() is used.

TODO:
load data in MNE and train on PCA / other models (needed for artifact rejection comparison)
"""

if __name__ == "__main__":
    """Process all sessions in a data folder and print the AUCs.
    
    This script is intended to be run from the command line with the following command:
        python offline_analysis_process_dataset.py
    """
    path = load_experimental_data()

    results = {}
    dropped = {}
    non_target = []
    target = []
    for session in Path(path).iterdir():
        if session.is_dir():
            # parameters = load_json_parameters(session / DEFAULT_PARAMETER_FILENAME, value_cast=True)
            # mne_data = mne.io.read_raw_fif(f'{session}/{ARTIFACT_LABELLED_FILENAME}')

            try:
                # grab the data and labels
                results[session.name] = {}

                # comment out training to use the same data for all sessions. Use ar_offline.py to train for CF.
                raw_data, data, labels, trigger_timing, channel_map, poststim_length, dt = load_data_inquiries(
                    data_folder=str(session.resolve())
                    )
                # epochs, figs = visualize_erp(
                #     raw_data,
                #     channel_map,
                #     trigger_timing,
                #     labels,
                #     poststim_length,
                #     transform=dt)
                # # average the epochs
                # non_target.append(epochs[0])
                # target.append(epochs[1])

                # train the models and get the results
                df = crossvalidate_record((data, labels), session_name=str(session.resolve()))
                for name in scores:
                    results[session.name][name] = df[f'mean_test_{name}']
                    results[session.name][f'std_{name}'] = df[f'std_test_{name}']
                # breakpoint()

                # dropped[session.name] = dl

            except Exception as e:
                print(f"Error processing session {session}: \n {e}")
                pass
    print(results)
    print(dropped)
    # file_name = 'cf_AAR_all_models.csv'
    # export = pd.DataFrame.from_dict(results).transpose()
    # export.to_csv(file_name)
    # export = pd.DataFrame.from_dict(dropped)
    # export.to_csv('cf_AAR_all_models_dropped.csv')

    # average across participants using grand_average
    # target_ga, non_target_ga = group_average(target, non_target)

    # add all the non-targets and targets to a list
    # epochs = non_target[0]
    # for nt in non_target[1:]: epochs += nt
    # visualize_evokeds((non_target, target), show=True)
    # visualize_joint_average((non_target, target), ['Non-Target', 'Target'], show=True)


    breakpoint()
    # final = df.copy()
    # for session, values in results.items():
    #     for name, value in values.items():
    #         values = np.array(values)
    #         final[session][f'mean_test_{name}'] = values.mean(axis=0)
    #         final[session][f'std_test_{name}'] = values.std(axis=0)
    # final.sort_values('mean_test_mcc', ascending=False)
    
    # import pdb; pdb.set_trace()
    # final.to_csv('full_run_OF_final.csv')
    import pdb; pdb.set_trace()