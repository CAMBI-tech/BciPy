"""Compare the results of the simulation for different paradigms (RSVP and Matrix).

This script is similar to the process mental effort script, but it will compare the paradigms instead of
examining the results of the simulation for a single paradigm.
"""
import json
from pathlib import Path
import pandas as pd
from typing import List
from bcipy.io.load import load_experimental_data
from scipy.stats import ttest_ind as t_test

from mental_effort_simulation import TRIAL_WINDOWS

from matplotlib import pyplot as plt
from matplotlib import cm


COLOR_MAP = cm.gist_earth

# Map of the summary data keys to the desired output keys.
# If the desired output key is None, then the value will be calculated using a custom formula.
SUMMARY_KEY_MAP = [
    ("N_SERIES", "total_number_series"),
    ("N_INQUIRIES", "total_inquiries"),
    ("N_SELECTIONS", "total_selections"),
    ("INQUIRIES_PER_SERIES", "inquiries_per_selection"),
    ("SELECTIONS_CORRECT", "task_summary__selections_correct"),
    ("SELECTIONS_INCORRECT", "task_summary__selections_incorrect"),
    ("SELECTIONS_CORRECT_SYMBOLS", "task_summary__selections_correct_symbols"),
    ("TYPING_ACCURACY", "task_summary__typing_accuracy"),
    ("TOTAL_SECONDS", "total_seconds"),
    ("CHAR_PER_MIN", None)
]
SUMMARY_FILE_NAME = "summary_data.json"
RESULTS_FILE_NAME = "group_results.csv"


def extract_sim_output(output_dir: Path) -> None:
    """Process the simulation output and generate a summary of the results."""
    
    output_data = {}
    for user in output_dir.iterdir():
        if user.is_dir():
            output_data[user.name] = {}
            for trial_window in user.iterdir():
                if trial_window.is_dir():
                    output_data[user.name][trial_window.name] = {}
                    for phrase in trial_window.iterdir():
                        if phrase.is_dir():
                            output_data[user.name][trial_window.name][phrase.name] = {}
                            for sim_run in phrase.iterdir():
                                if sim_run.is_dir():
                                    summary_data_path = sim_run / SUMMARY_FILE_NAME
                                    summary_data = summary_stats_for_sim_output(summary_data_path)
                                    output_data[user.name][trial_window.name][phrase.name]= summary_data
    return output_data
                                    

def summary_stats_for_sim_output(summary_data_path: Path) -> dict:
    """Generate summary statistics for the simulation output.
    
    Parameters
    ----------
    summary_data_path : Path
        The path to the summary data file. This should be a JSON file containing the summary data for the simulation run.
    """
    # load json summary as a dictionary
    with open(summary_data_path, "r") as f:
        summary_data = json.load(f)
    
    # map the keys to the summary data
    summary_stats = {}
    for value, key in SUMMARY_KEY_MAP:
        if key:
            data: List[float] = summary_data[key]
            average = sum(data) / len(data)
            std_dev = sum((x - average) ** 2 for x in data) / len(data)

            summary_stats[f"{value}_AVG"] = average
            summary_stats[f"{value}_STD"] = std_dev
        elif value == "CHAR_PER_MIN":
            # Get the average length of typed text and divide by the average time to get characters per minute
            typed_text: List[str] = summary_data["typed"]
            total_seconds: List[float] = summary_data["total_seconds"]
            avg_typed_length = sum(len(text) for text in typed_text) / len(typed_text)
            std_typed_length = sum((len(text) - avg_typed_length) ** 2 for text in typed_text) / len(typed_text)
            avg_total_seconds = sum(total_seconds) / len(total_seconds)
            avg_total_minutes = avg_total_seconds / 60
            avg_chars_per_min = avg_typed_length / avg_total_minutes
            std_chars_per_min =  sum(
                (len(text) / (time / 60) - avg_chars_per_min) ** 2 for text, time in zip(typed_text, total_seconds)) / len(typed_text)
            summary_stats[f"{value}_AVG"] = avg_chars_per_min
            summary_stats[f"{value}_STD"] = std_chars_per_min
        else:
            raise ValueError(f"Key is None for value {value}. Only the custom CHAR_PER_MIN value should have a None key.")

    return summary_stats

def save_results(data: dict, output_dir: Path, mode='RSVP') -> pd.DataFrame:
    """Save the results to the output directory as a CSV file.
    
    Data should be a dictionary with the following structure:

    ```python
    {
        "user1": {
            "trial_window1": {
                    "N_SERIES_AVG": 10,
                    "N_SERIES_STD": 2,
                    "N_INQUIRIES_AVG": 5,
                    "N_INQUIRIES_STD": 1,
                    ...
                },
            },
            "trial_window2": {
                ...
            }
        },
        "user2": {
            ...
        }
    }
    """
    df = pd.DataFrame(data)
    # this gives us cols for each trial window and rows for each user
    df = df.transpose()
    # process the data such that several columns are created for each trial window based on the keys in the dictionary
    # The first column should be trial_window_metric, the second should be trial_window_metric_STD
    # and so on for each metric in the dictionary.

    reference_frame = df.copy()
    for trial_window, trial_data in reference_frame.items():
        # breakpoint()
        for user, metrics in trial_data.items():
            for metric, value in metrics.items():
                new_col_name = f"{trial_window}_{metric}"

                df.at[user, new_col_name] = value
                
    
    # drop the original columns
    df = df.drop(columns=reference_frame.columns.tolist())
    filename = f"{mode}_{RESULTS_FILE_NAME}"
    df.to_csv(output_dir / filename)
    return df


def process_sim_output(data: dict) -> dict:
    """Process the simulation output data to average the results for each language model across phrases.

    Output should be a dictionary with the following structure:

    {
        "user1": {
            "trail_window1": {
                "N_SERIES_AVG": 10,
                "N_SERIES_STD": 2,
                "N_INQUIRIES_AVG": 5,
                "N_INQUIRIES_STD": 1,
                ...
            },
            "trial_window2": {
                ...
            }
        },
        "user2": {
            ...
    }
    """
    # Loop over the phrases for each language model and calculate the average and standard deviation for each metric
    final_data = {}
    for user, trial_windows_dir in data.items():
        final_data[user] = {}
        for trial_window_dir, phrases in trial_windows_dir.items():
            trial_window = trial_window_dir.split("/")[-1]
            trial_window_str = trial_window.replace("_", ":")
            final_data[user][trial_window] = {}
            if trial_window_str in TRIAL_WINDOWS:
                phrase_data = {}
                for _phrase, summary_data in phrases.items():
                    for key, value in summary_data.items():
                        if key not in phrase_data:
                            phrase_data[key] = []
                        phrase_data[key].append(value)
                # Calculate the average and standard deviation for each metric
                for key, value in phrase_data.items():
                    average = sum(value) / len(value)
                    std_dev = sum((x - average) ** 2 for x in value) / len(value)
                    final_data[user][trial_window][f'{key}'] = average
                    final_data[user][trial_window][f'{key}_STD']  = std_dev
            else:
                print(f"Skipping language model {trial_window} as it is not in the list of valid language models")
    return final_data

def grab_data_keys(metrics: List[str], models: List[str]) -> List[str]:
    """Generate the data keys for the specified metrics and models."""
    data_keys = []
    for model in models:
        model = model.replace(":", "_")
        for metric in metrics:
            key = f"{model}_{metric}"
            data_keys.append(key)
    return data_keys

def plot_results(
        data: dict,
        metric: str,
        data_keys: List[str],
        output_dir: Path,
        show=True,
        save=True) -> None:
    """Plot the results of the simulation.
    
    The data should be a dictionary with the following structure:
        {'RSVP': dataframe, 'Matrix': dataframe}
    
    Metrics should be a list of the metrics to plot.
    Data keys should be a list of the data keys to plot from the dataframes

    We want to compare the results of each metric across modes (RSVP or Matrix).
    """

    # grab matrix and rsvp dataframes average for each metric
    rsvp_data = data["RSVP"].copy()
    matrix_data = data["Matrix"].copy()

    rsvp_data = rsvp_data[data_keys].mean(axis=0)
    matrix_data = matrix_data[data_keys].mean(axis=0)

    rsvp_data_std = data["RSVP"][data_keys].std(axis=0) / (len(data["RSVP"]) ** 0.5)
    matrix_data_std = data["Matrix"][data_keys].std(axis=0) / (len(data["Matrix"]) ** 0.5)

    trial_labels = [f'{key.split("_")[0]}:{key.split("_")[1]}' for key in data_keys]
    plt.errorbar(trial_labels, rsvp_data, yerr=rsvp_data_std, label='RSVP', fmt='o')
    plt.errorbar(trial_labels, matrix_data, yerr=matrix_data_std, label='Matrix', fmt='o')

    plt.title(f"{metric} Comparison")
    plt.xlabel("Trial Window")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if save:
        plt.savefig(output_dir / f"{metric}_comparison.png")
    if show:
        plt.show()

    pass

def run_stats(
        dist_1: List[float],
        dist_2: List[float],
        key_1: str,
        key_2: str,
        significant_results: dict,
        results: dict,
        permutations=10000) -> None:
    """Run statistical tests on the results.
    
    This function should run a series of statistical tests to determine if there is a significant
        difference between the keys for the specified metrics.
    """
    # Run a t-test
    t_stat, p_value = t_test(dist_1, dist_2, permutations=permutations, alternative="two-sided")
    # print(f"t-statistic: {t_stat}, p-value: {p_value}")
    results[f"{key_1} vs {key_2}"] = (t_stat, p_value)
    if p_value < 0.05:
        significant_results[f"{key_1} vs {key_2}"] = (t_stat, p_value)
        print(f"Significant difference between {key_1} and {key_2}: t-statistic = {t_stat}, p-value = {p_value}")

    return results, significant_results

if __name__ == "__main__":
    modes = ["RSVP", "Matrix"]
    data_path = Path(load_experimental_data(message="Select the directory containing the simulation output"))

    data_by_mode = {}

    for sub_path in data_path.iterdir():
        mode = sub_path.name
        if mode not in modes:
            continue
        data = extract_sim_output(Path(sub_path))
        final_data = process_sim_output(data)
        df = save_results(final_data, Path(sub_path), mode=mode)
        data_by_mode[mode] = df

    # for plotting and statistical analysis
    processing_metrics = [
        "TOTAL_SECONDS_AVG",
        "CHAR_PER_MIN_AVG",
        "N_INQUIRIES_AVG",
        "TYPING_ACCURACY_AVG",
        "SELECTIONS_INCORRECT_AVG",
        ]
    # processing_metrics = ["TYPING_ACCURACY_AVG"]
    # processing_metrics = ["N_SERIES_AVG"]
    processing_models = TRIAL_WINDOWS

    for metric in processing_metrics:
        data_keys = grab_data_keys([metric], processing_models)
        plot_results(data_by_mode, metric, data_keys, data_path, show=True, save=True)

    # loop over matrix and rsvp dataframes and run stats on each metric
    results = {}
    significant_results = {}
    for metric in processing_metrics:
        data_keys = grab_data_keys([metric], processing_models)
        for i, key_1 in enumerate(data_keys):
            dist_1 = data_by_mode["RSVP"][key_1].tolist()
            dist_2 = data_by_mode["Matrix"][key_1].tolist()
            key_1 = f"RSVP_{key_1}"
            key_2 = f"Matrix_{key_1}"
            results, significant_results = run_stats(dist_1, dist_2, key_1, key_2, significant_results, results)

    breakpoint()