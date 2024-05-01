"""Offline Analysis - Process Dataset

Process all sessions in a data folder and export the results to a csv file.
    
    This script is intended to be run from the command line with the following command:
        python offline_analysis_process_dataset.py

    To change the models or training procedures used, edit the clfs dictionary in grid_search_classifier.py.
"""

from pathlib import Path

from bcipy.helpers.process import load_data_inquiries, load_data_mne
from bcipy.helpers.load import load_experimental_data, load_json_parameters
from grid_search_classifier import crossvalidate_record, scores
from bcipy.config import DEFAULT_PARAMETERS_PATH

import mne
from mne import grand_average
import pandas as pd
from tqdm import tqdm

def group_average(target_evoked, nontarget_evoked):
    """Given a list of target and nontarget evokeds, return a grand average of both."""
    target_evoked = grand_average(target_evoked)
    nontarget_evoked = grand_average(nontarget_evoked)
    return target_evoked, nontarget_evoked

# If labelled artifact data in dataset exists, define the filename here
ARTIFACT_LABELLED_FILENAME = 'artifacts_raw.fif'
# FILTER_NOISE = ['BAD_blink', 'BAD_emg'] # subset of labels
FILTER_NOISE = ['BAD_blink', 'BAD_emg', 'BAD_event', 'BAD_eog'] # all labels


if __name__ == "__main__":

    path = load_experimental_data()

    results = {}
    dropped = {}
    non_target = []
    target = []

    parameters = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True)
    # parameters = None # use this to use the parameters in the data folder (not the default)
    
    progress_bar = tqdm(
        Path(path).iterdir(),
        total=len(list(Path(path).iterdir())),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [est. {remaining}][ela. {elapsed}]\n",
        colour='MAGENTA')

    for session in progress_bar:                               
        if session.is_dir():
            session_folder = str(session.resolve())

            try:
                # grab the data and labels
                results[session.name] = {}
                progress_bar.set_description(f"Processing {session.name}...")

                # # uncomment to use the inquiry based filtering
                raw_data, data, labels, trigger_timing, channel_map, poststim_length, default_transform, dl, _ = load_data_inquiries(
                    data_folder=session_folder
                    )

                # create epochs from inquiry data
                # chanbel_type = 'eeg' * len(channels)
                # info = mne.create_info(ch_names=channels, sfreq=fs)
                # epochs = mne.EpochsArray(data)
                
                # # If using artifact labelled data, or whole dataset filtering, uncomment these lines
                # mne_data = mne.io.read_raw_fif(f'{session}/{ARTIFACT_LABELLED_FILENAME}')
                # all_annotations = mne_data.annotations
                # new_annotation_onset = []
                # new_annotation_duration = []
                # new_annotation_description = []
                # for t in all_annotations:
                #     if t['description'] in FILTER_NOISE:
                #         new_annotation_onset.append(t['onset'])
                #         new_annotation_duration.append(t['duration'])
                #         new_annotation_description.append('BAD_noise')
                
                # filtered_annotations = mne.Annotations(
                #     new_annotation_onset,
                #     new_annotation_duration,
                #     new_annotation_description)
                # raw_data, data, labels, trigger_timing, channel_map, poststim_length, default_transform, dl, _  = load_data_mne(
                #     data_folder=session_folder, drop_artifacts=False, parameters=parameters)

                # train the models and get the results. Provide a session name, to persist the models.
                df = crossvalidate_record((data, labels))
                for name in scores:
                    results[session.name][name] = df[f'mean_test_{name}']
                    results[session.name][f'std_{name}'] = df[f'std_test_{name}']

                dropped[session.name] = dl

            except Exception as e:
                print(f"Error processing session {session}: \n {e}")
                raise e
    
    # export the results!
    # lp only = NAR_lponly_real_RetrainwithMDM (High pass of 1 Hz). Miss named!
    # cf_real = NAR_CF_real_RetrainwithMDM (1-20 Hz)  CF
    # of = NAR_OF_MDM_3_cov (1-20 Hz) OF
    # of_no_buffer = OF but no transformation buffer to prove the edge effects hypothesis
    # TODO: of hp only, check results for regular filter, try .1-75 Hz?
    condition = 'NAR_OF_RetrainwithMDM_limited'
    file_name = f'{condition}_all_models.csv'
    
    export = pd.DataFrame.from_dict(results).transpose()
    export.to_csv(file_name)

    drop_export = pd.DataFrame.from_dict(dropped)
    drop_export.to_csv(f'{condition}_all_models_dropped.csv')

    progress_bar.close()
    print("Done!")