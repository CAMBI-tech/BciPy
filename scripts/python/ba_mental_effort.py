
from pathlib import Path
from tqdm import tqdm
import os
from bcipy.signal.model.offline_analysis import offline_analysis
from bcipy.io.load import load_json_parameters, load_experimental_data
import pandas as pd
from bcipy.helpers.process import load_data_inquiries
from .grid_search_classifier import crossvalidate_record, scores



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
        session_folder = str(session.resolve())

        try:
            # grab the data and labels
            results[session.name] = {}
            progress_bar.set_description(f"Processing {session.name}...")
            parameters = load_json_parameters(f"{session}/parameters.json", value_cast=True)

            model_path = f'{path}/models/{session.name}'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            # Train the model
            df = train_model(session_folder, parameters, model_path)
            results[session.name]['df'] = df
            for name in scores:
                results[session.name][name] = df[f'mean_test_{name}']
                results[session.name][f'std_{name}'] = df[f'std_test_{name}']



        except Exception as e:
            print(f"Error processing {session.name}: {e}")
            breakpoint()
            continue

filename = 'retrain_ME_results.csv'
export = pd.DataFrame.from_dict(results).transpose()
export.to_csv(filename)

progress_bar.close()
breakpoint()
