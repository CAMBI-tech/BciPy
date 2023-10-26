from bcipy.helpers.visualization import visualize_erp, visualize_evokeds, visualize_joint_average
from bcipy.helpers.process import load_data_inquiries, load_data_trials, load_data_mne
from bcipy.helpers.load import load_experimental_data, load_json_parameters
from soa_classifier import crossvalidate_record, scores
import warnings
warnings.filterwarnings('ignore')
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
from tqdm import tqdm


# ARTIFACT_LABELLED_FILENAME = 'artifacts_raw.fif'
ARTIFACT_LABELLED_FILENAME = 'auto_artifacts_raw.fif'

"""
Processing notes:

- OF/IB: The data is loaded, reshaped to inquiries, filtered, then reshaped to trials. load_data_inquiries() is used.
- CF/WD: The data is loaded, filtered, then reshaped to trials. load_data_trials() is used.
- NAR: no artifact rejection is performed
- AAR: automated artifact rejection is performed
- SAR: semi-automated artifact rejection is performed
- IIR: IIR filtering is performed
- FIR: FIR filtering is performed

TODO:

+ ensure the AAR is AAR and not SAR; From the old spreadsheet it was SAR
+ check how the cohen's kappa is calculated and write up the formula

+ input into spreadsheet
+ calculate the t-tests for all observations
+ generate the graphs for all observations
+ generate the graphs for the grand averages
+ write up paper D;
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
    progress_bar = tqdm(
        Path(path).iterdir(),
        desc="Processing Artifact Dataset... \n",
        total=len(list(Path(path).iterdir())),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [est. {remaining}][ela. {elapsed}]\n", # can I do math in this and calculate iteration average time?
        colour='green')
    for session in progress_bar:                               
        if session.is_dir():
            # parameters = load_json_parameters(session / DEFAULT_PARAMETER_FILENAME, value_cast=True)
            # mne_data = mne.io.read_raw_fif(f'{session}/{ARTIFACT_LABELLED_FILENAME}')
            session_folder = str(session.resolve())

            try:
                # grab the data and labels
                results[session.name] = {}
                progress_bar.set_description(f"Processing {session.name}...")

                # comment out training to use the same data for all sessions. Use ar_offline.py to train for CF.
                raw_data, data, labels, trigger_timing, channel_map, poststim_length, default_transform, dl = load_data_inquiries(
                    data_folder=session_folder
                    )
                
                # If using artifact labelled data, uncomment these lines
                # mne_data = mne.io.read_raw_fif(f'{session}/{ARTIFACT_LABELLED_FILENAME}')
                # raw_data, data, labels, trigger_timing, channel_map, poststim_length, default_transform, dl  = load_data_mne(
                #     data_folder=session_folder, mne_data_annotations=mne_data.annotations, drop_artifacts=True)
     

                # train the models and get the results
                df = crossvalidate_record((data, labels), session_name=str(session.resolve()))
                for name in scores:
                    results[session.name][name] = df[f'mean_test_{name}']
                    results[session.name][f'std_{name}'] = df[f'std_test_{name}']

                dropped[session.name] = dl

            except Exception as e:
                print(f"Error processing session {session}: \n {e}")
                pass
    
    # export the results!
    condition = 'OF_NAR_IIR'
    file_name = f'{condition}_all_models.csv'
    export = pd.DataFrame.from_dict(results).transpose()
    export.to_csv(file_name)
    export = pd.DataFrame.from_dict(dropped)
    export.to_csv(f'{condition}_all_models_dropped.csv')

    import pdb; pdb.set_trace()
    progress_bar.close()