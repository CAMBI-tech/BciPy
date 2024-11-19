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
                # raw_data, data, labels, trigger_timing, channel_map, poststim_length, default_transform, device_spec = load_data_trials(
                #     session_folder,
                #     parameters=parameters)

                # train the models and get the results. Provide a session name, to persist the models.
                # df = crossvalidate_record((data, labels))
                # for name in scores:
                #     results[session.name][name] = df[f'mean_test_{name}']
                #     results[session.name][f'std_{name}'] = df[f'std_test_{name}']

                # dropped[session.name] = dl

                # Train and save the PCA model trained using the data
                model = train_bcipy_model(
                    data, labels, session_folder, device_spec, default_transform, 'condition')
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
CF-2: [0.8609203262668609, 0.48354294888948357, 0.8913666141388913, 0.7915539301677914, 0.8088779906961726, 0.8467477378368468, 0.8903656131378903, 0.8703059495138702, 0.7563008553107562, 0.8506229001278506, 0.9573335711949573, 0.5305107087285306, 0.9245483106869246, 0.7417813853457417, 0.7878076095897878, 0.908026838719908, 0.8240121309428241, 0.8858561531828859, 0.6234947819106235, 0.8436059822198436, 0.793040565317793, 0.9664813328179664, 0.7941902298337942, 0.8446069832208446, 0.9354899453909354, 0.5378745081715378, 0.7920494752177921, 0.8311281578608312, 0.7582136592037582, 0.8232291697638233]
CF: [0.8573821346098575, 0.5639599996035639, 0.8799393452858799, 0.8034073677638034, 0.7992916174734358, 0.8237048930118237, 0.8790077205918791, 0.825934845736826, 0.7093529172737094, 0.857064985777857, 0.9513572978919513, 0.49521799026749524, 0.9099495535139099, 0.7396108980267397, 0.7567864894597568, 0.8993746221468994, 0.8228030010208227, 0.8728134074668729, 0.5946738818025946, 0.8255384096968256, 0.7856173004687856, 0.9552721037869552, 0.764051179892764, 0.8393244729878393, 0.9236563295969237, 0.537022170685537, 0.7787292242737788, 0.772108742405772, 0.7543682296157544, 0.8286405217098287]
CF-8: [0.8497705626418498, 0.5438309596725438, 0.8897412263748897, 0.7883923527487884, 0.7960322505777051, 0.8317822773268317, 0.8834577151408836, 0.8621790106938623, 0.7484514217187485, 0.8354790433998354, 0.9572146403829572, 0.5213629471055214, 0.9166493225899167, 0.7226335246137227, 0.7630204461887629, 0.8987799680868989, 0.8225849611988225, 0.8782743139178782, 0.6125432363056126, 0.8457070932318457, 0.7862416872317862, 0.9612979315949614, 0.7770939256087771, 0.838987502353839, 0.9260151240349259, 0.5168534871505168, 0.79013667132479, 0.8182737192638183, 0.7567072022517568, 0.8180854121448182]
OF-2: [0.860920326266861, 0.48702167514048705, 0.8928631601898929, 0.7919008117027919, 0.8126015398742672, 0.8461530837768461, 0.8922982388328923, 0.8659154203708659, 0.7527131091487527, 0.8556179942318555, 0.9586120774239585, 0.5432660383155432, 0.9258069951139257, 0.740938958760741, 0.7908007016917907, 0.9070753922239071, 0.8249338447358251, 0.8873725210358873, 0.6321767311866322, 0.8449538647558451, 0.7927828818917929, 0.9671750958879671, 0.7943190715467944, 0.8462224600838463, 0.9334780324879335, 0.48811187425048813, 0.7927630600897928, 0.8243094579728242, 0.7635060803377635, 0.8216136929008216]
OF: [0.8494335920078495, 0.5302728471045303, 0.8874914518478875, 0.7862119545287863, 0.8056388056388055, 0.8352808253798353, 0.8869265304908869, 0.8541412699828542, 0.7512264739987512, 0.841861663643842, 0.9584138594039584, 0.543979623187544, 0.9198703654149198, 0.7232579113767232, 0.763010535287763, 0.9016045748719017, 0.8341509826658342, 0.8808115045738808, 0.6236632672276237, 0.8428329319418428, 0.791058385117791, 0.9624376852099624, 0.7732980505257733, 0.840959771652841, 0.9265800453919265, 0.5366852000515366, 0.79001774051279, 0.8110189397318109, 0.7598786905717599, 0.8237544475168239]
OF-8: [0.8487299180368487, 0.5261697340905261, 0.8871247485108872, 0.7924756439607925, 0.8000181636545273, 0.8330607835558331, 0.8823972487338824, 0.8504048603058504, 0.7498587696607499, 0.836886391341837, 0.9604059505049605, 0.5488756082815489, 0.915004113023915, 0.7170140437467171, 0.7579163321737579, 0.8943299735378943, 0.8192053439578193, 0.878016630491878, 0.6154372193976154, 0.8364899553018366, 0.7825151884557825, 0.9588895826519589, 0.7646755666557646, 0.8413661185938414, 0.9239833893299241, 0.5318387694625317, 0.7851514881217851, 0.7898492551957899, 0.7516129991377516, 0.8187593534128188]
"""