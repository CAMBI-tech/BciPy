
from pathlib import Path
from tqdm import tqdm
import os
from bcipy.signal.model.offline_analysis import offline_analysis
from bcipy.io.load import load_json_parameters, load_experimental_data
import pandas as pd
from bcipy.helpers.process import load_data_inquiries
from grid_search_classifier import crossvalidate_record, scores



def train_model(data_path, parameters, model_path):
    """
    Train a model using the given data and parameters.

    Args:
        data_path (str): Path to the data file.
        parameters (dict): Parameters for training the model.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Load the data
    raw_data, data, labels, trigger_timing, channel_map, poststim_length, default_transform, dl, _, device_spec = load_data_inquiries(
                    data_folder=data_path,
                    parameters=parameters,
                    apply_filter=True,
                )
    # Train the model
    df = crossvalidate_record((data, labels), session_name=model_path)
    return df



path = load_experimental_data()
progress_bar = tqdm(
        Path(path).iterdir(),
        total=len(list(Path(path).iterdir())),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [est. {remaining}][ela. {elapsed}]\n",
        colour='MAGENTA')
results = {}
results['all'] = []
for session in progress_bar:                               
    if session.is_dir():
        for session_sub in session.iterdir():
            # skip non-directories
            if not session_sub.is_dir():
                continue
            session_folder = str(session_sub.resolve())

            print(f"Processing {session_sub.name}...")
            # check if this is a calibration session
            if 'calibration' in session_sub.name.lower():
                try:
                    # grab the data and labels
                    results[session_sub.name] = {}
                    progress_bar.set_description(f"Processing {session_sub.name}...")
                    parameters = load_json_parameters(f"{session_sub}/parameters.json", value_cast=True)

                    model_path = f'./retrained_models/{session_sub.name}'
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                                        # Train the model
                    df = train_model(session_folder, parameters, model_path)
                    results[session_sub.name]['df'] = df
                    for name in scores:
                        results[session_sub.name][name] = df[f'mean_test_{name}']
                        results[session_sub.name][f'std_{name}'] = df[f'std_test_{name}']



                except Exception as e:
                    print(f"Error processing {session_sub.name}: {e}")
                    breakpoint()
                    continue
            else:
                progress_bar.set_description(f"Skipping {session.name}...")
                continue

filename = './retrained_models/retrain_ME_results.csv'
print(f"Saving results to {filename}...")

# Save the results to a CSV file
if os.path.exists(filename):
    os.remove(filename)

breakpoint()

export = pd.DataFrame.from_dict(results).transpose()
export.to_csv(filename)

progress_bar.close()
breakpoint()
