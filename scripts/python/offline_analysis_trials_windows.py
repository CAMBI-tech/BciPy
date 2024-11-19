"""Offline Analysis - Trial Windows Analysis

This script loads a participant's data and retrains models using varied trial window sizes, then save and plot the results.
It will also save the model to the provided data path.

Usage:

    `python offline_analysis_trials_windows.py`

    To see all available options and defaults, use the --help flag:

        `python offline_analysis_trials_windows.py --help`

    To change the output path, which defaults to the current working directory, use the --output flag:

        `python offline_analysis_trials_windows.py --output <output_path>`

    To change the trial windows to test, use the --window flag for each window size you want to test:
        
        `python offline_analysis_trials_windows.py --window "start_time end_time" --window "start_time end_time"`
"""
import os
from typing import List
from bcipy.signal.model import SignalModel
from bcipy.signal.model.offline_analysis import offline_analysis
from bcipy.helpers.parameters import Parameters
import pandas as pd
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the window sizes to test
TRIAL_WINDOWS: List[tuple] = [(0, 0.5), (0.2, 0.7)]
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
# TRIAL_WINDOWS: List[tuple] = [
#     (0, 0.3), (0, 0.5), (0, 0.7),
#     (0.1, 0.4), (0.1, 0.6), (0.1, 0.8),
#     (0.2, 0.5), (0.2, 0.7),
#     (0.3, 0.6), (0.3, 0.8),
#     (0.4, 0.7),
#     (0.5, 0.8)
#     ]

def save_output_csv(data: pd.DataFrame, output_path: Path) -> None:
    """Save the output data to a csv file.
    
    Args:
        data: The data to save.
            Expected format: {'trial_window': auc}
        output_path: The path to save the file to
    
    Returns:
        None
    """
    csv_path = os.path.join(output_path, f'trial_window_analysis_{TIMESTAMP}.csv')
    data.to_csv(csv_path, index=False, header=True)


def plot_output_results(data: pd.DataFrame, output_path: Path) -> None:
    """Plot the output data to a bar chart.
    
    Args:
        data: The data to plot.
            Expected format: {'trial_window': auc}
        output_path: The path to save the file to
    
    Returns:
        None
    """
    df = data.T
    df.columns = ['AUC']
    df = df.reset_index()
    df = df.rename(columns={'index': 'Trial Window'})
    df = df.sort_values(by='AUC', ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Trial Window', y='AUC', data=df, palette='viridis')
    plt.title('AUC by Trial Window')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_path, f'trial_window_analysis_{TIMESTAMP}.png'))


def train_default_bcipy_model(data_path: Path, parameters: Parameters) -> SignalModel:
    """Train a model using the provided data and parameters.
    
    Args:
        data_path: The path to the bcipy data folder to use for the analysis.
        parameters: The parameters to use for the analysis.
    
    Returns:
        SignalModel: The trained model
    """
    model = offline_analysis(
        data_path,
        parameters,
        alert_finished = False,
        estimate_balanced_acc = False,
        show_figures = False,
        save_figures = False,
    )
    assert len(model) == 1, "Expected a single model to be returned."
    return model[0]

def main(data_path: Path, parameters: Parameters, output_path: Path, windows: List[tuple]) -> None:
    """Main function to run the analysis.
    
    Args:
        data_path: The path to the bcipy data folder to use for the analysis.
        parameters: The parameters to use for the analysis.
            The trial_window parameter will be updated for each window size defined in TRIAL_WINDOWS.
        output_path: The path to save the output data to.
        windows: The trial windows to test. Each window should be a tuple of (start_time, end_time).
    """
    output_data = {}

    # Train a model for each window size
    for window in windows:
        output_data[window] = {}
        # Update the parameters with the current window
        parameters['trial_window'] = window

        model = train_default_bcipy_model(data_path, parameters)
        output_data[window] = model.auc
    
    # Print the output data to the console
    print(output_data)
    # Save the output data to a csv file and plot the results
    df = pd.DataFrame(output_data)
    save_output_csv(df, output_path)
    plot_output_results(df, output_path)
    

if __name__ in "__main__":
    import argparse
    from bcipy.helpers.load import load_experimental_data, load_json_parameters
    from bcipy.config import DEFAULT_PARAMETERS_FILENAME

    parser = argparse.ArgumentParser(
        prog="offline_analysis_process_windows",
        description="Offline Analysis - Process Windows",
        epilog="This script loads a participant's data and retrains models using varied window sizes, then save and plot the results.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=False,
        help="Output path for the results. Default is the current working directory.",
        default=os.getcwd())
    parser.add_argument(
        "--window",
        "-w",
        type=str,
        action="append",
        required=False,
        metavar="start_time end_time",
        help=(
            f"The trial windows to test. Defaults are {TRIAL_WINDOWS}. "
             "Use this flag to override the defaults. "
             "Multiple windows can be specified by repeating the flag.")
    )
    
    args = parser.parse_args()
    windows = args.window
    # Parse the windows into a list of tuples
    if windows:
        windows = [tuple(map(float, window.split())) for window in windows]
    else:
        windows = TRIAL_WINDOWS
    path = load_experimental_data()

    parameter_path = os.path.join(path, DEFAULT_PARAMETERS_FILENAME)
    if not os.path.exists(parameter_path):
        raise FileNotFoundError(f"Parameters file not found at {parameter_path}")

    parameters = load_json_parameters(parameter_path, value_cast=True)
    main(Path(path), parameters, Path(args.output), windows)
