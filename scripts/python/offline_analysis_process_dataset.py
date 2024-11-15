"""Offline Analysis - Process Dataset

Process all sessions in a data folder and export the results to a csv file.
    
    This script is intended to be run from the command line with the following command:
        python offline_analysis_process_dataset.py

    To change the models or training procedures used, edit the clfs dictionary in grid_search_classifier.py.
"""

from pathlib import Path

from bcipy.helpers.process import load_data_inquiries, load_data_mne, load_data_trials
from bcipy.helpers.load import load_experimental_data, load_json_parameters
from grid_search_classifier import crossvalidate_record, scores
from bcipy.config import DEFAULT_PARAMETERS_PATH
from train_bcipy_model import train_bcipy_model

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

    # results = {}
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
    results = {}
    results['all'] = []
    for session in progress_bar:                               
        if session.is_dir():
            session_folder = str(session.resolve())

            try:
                # grab the data and labels
                results[session.name] = {}
                progress_bar.set_description(f"Processing {session.name}...")

                # # # # uncomment to use the inquiry based filtering
                raw_data, data, labels, trigger_timing, channel_map, poststim_length, default_transform, dl, _, device_spec = load_data_inquiries(
                    data_folder=session_folder,
                    parameters=parameters
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

                # Whole dataset filter
                # raw_data, data, labels, trigger_timing, channel_map, poststim_length, default_transform, dl, _, device_spec  = load_data_mne(
                #     data_folder=session_folder, drop_artifacts=False, parameters=parameters)
                # raw_data, data, labels, trigger_timing, channel_map, poststim_length, default_transform, device_spec = load_data_trials(session_folder)

                # train the models and get the results. Provide a session name, to persist the models.
                # df = crossvalidate_record((data, labels))
                # for name in scores:
                #     results[session.name][name] = df[f'mean_test_{name}']
                #     results[session.name][f'std_{name}'] = df[f'std_test_{name}']

                # dropped[session.name] = dl

                # Train and save the PCA model trained using the data
                model = train_bcipy_model(data, labels, session_folder, device_spec, default_transform)
                results[session.name]['AUC'] = model.auc
                results['all'].append(model.auc)
            except Exception as e:
                print(f"Error processing session {session}: \n {e}")
                raise e
    
    # export the results!
    # lp only = NAR_lponly_real_RetrainwithMDM (High pass of 1 Hz). Miss named!
    # cf_real = NAR_CF_real_RetrainwithMDM (1-20 Hz)  CF
    # of = NAR_OF_MDM_3_cov (1-20 Hz) OF
    # of_no_buffer = OF but no transformation buffer to prove the edge effects hypothesis
    # TODO: of hp only, check results for regular filter, try .1-75 Hz?
    # condition = 'NAR_OF_1_50Hz'
    # file_name = f'{condition}_all_models.csv'
    
    # export = pd.DataFrame.from_dict(results).transpose()
    # export.to_csv(file_name)

    # drop_export = pd.DataFrame.from_dict(dropped)
    # drop_export.to_csv(f'{condition}_all_models_dropped.csv')

    progress_bar.close()
    print("Done!")
    breakpoint()
    # save the results dictionary


"""
CF: [0.8573821346098575, 0.5639599996035639, 0.8799393452858799, 0.8034073677638034, 0.7992916174734358, 0.8237048930118237, 0.8790077205918791, 0.825934845736826, 0.7093529172737094, 0.857064985777857, 0.9513572978919513, 0.49521799026749524, 0.9099495535139099, 0.7396108980267397, 0.7567864894597568, 0.8993746221468994, 0.8228030010208227, 0.8728134074668729, 0.5946738818025946, 0.8255384096968256, 0.7856173004687856, 0.9552721037869552, 0.764051179892764, 0.8393244729878393, 0.9236563295969237, 0.537022170685537, 0.7787292242737788, 0.772108742405772, 0.7543682296157544, 0.8286405217098287]
OF: [0.8494335920078495, 0.5302728471045303, 0.8874914518478875, 0.7862119545287863, 0.8056388056388055, 0.8352808253798353, 0.8869265304908869, 0.8541412699828542, 0.7512264739987512, 0.841861663643842, 0.9584138594039584, 0.543979623187544, 0.9198703654149198, 0.7232579113767232, 0.763010535287763, 0.9016045748719017, 0.8341509826658342, 0.8808115045738808, 0.6236632672276237, 0.8428329319418428, 0.791058385117791, 0.9624376852099624, 0.7732980505257733, 0.840959771652841, 0.9265800453919265, 0.5366852000515366, 0.79001774051279, 0.8110189397318109, 0.7598786905717599, 0.8237544475168239]
"""