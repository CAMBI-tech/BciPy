"""Process Sim Output.

This script processes the output of the simulation and generates a summary of the results. The current output directory structure is as follows:

```text
output_dir/
    user1/
        trail_window1/
            phrase1/
                user1_phrase1_trail_window1_SIM_datetime/
                    run_1/
                        run_1.log
                        session.json
                        session.xlsx
                        triggers.txt
                    run_2/
                        ...
                    summary_data.json
                    metrics.png
            phrase2/
                ...
        trial_window2/
            phrase1/
                user1_phrase1_trial_window2_SIM_datetime/
                    ...
    user2/
        ...
```

The data will be processed and averages extracted for each language model across phrases. The results will be saved in a CSV file in the output directory.

TODO: review with Dylan. 
- Determine best way to estimate characters per minute per phrase
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
        data: pd.DataFrame,
        metric: str,
        data_keys: List[str],
        output_dir: Path,
        mode='RSVP',
        show=True,
        save=True) -> None:
    """Plot the results of the simulation."""
    # BAR PLOT
    len_bars = len(data_keys)
    colors = dict(zip(data_keys, [COLOR_MAP(i / len_bars) for i in range(len(data_keys))]))
    plot = data[data_keys].plot(kind="bar", figsize=(10, 6), color=colors)
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.xlabel("User")
    plt.title(f"{metric} by Trial Window")

    if show:
        plt.show()
    
    if save:
        fig = plot.get_figure()
        fig.savefig(f"{output_dir}/{mode}_{metric}_bar.png")

    colors = dict(boxes="#6da066", whiskers="#0f0f0f", medians="#0f0f0f", caps="#0f0f0f")
    # BOXPLOT
    boxplot = data.boxplot(column=data_keys, figsize=(10, 6), fontsize=15, color=colors, patch_artist=True)
    plt.ylabel(metric)
    label_text = [f"{key.split('_')[0]}:{key.split('_')[1]}" for key in data_keys]
    plt.xticks(range(1, len(data_keys) + 1), label_text)

    if show:
        plt.show()

    if save:
        fig = boxplot.get_figure()
        fig.savefig(f"{output_dir}/{mode}_{metric}_box.png")

    # average all the users for each the metric and trial window
    # and plot the average for each trial window
    avg_data = data[data_keys].mean(axis=0)
    avg_data = pd.DataFrame(avg_data).transpose()
    colors = dict(zip(label_text, [COLOR_MAP(i / len_bars) for i in range(len(label_text))]))
    avg_data = avg_data.rename(columns={key: f'{key.split("_")[0]}:{key.split("_")[1]}' for key in data_keys})
    plot = avg_data.plot(kind="bar", figsize=(10, 6), color=colors)
    plt.ylabel(metric)
    plt.xlabel("Trial Window")
    plt.title(f"{metric} by Trial Window (Average)")

    if show:
        plt.show()
    
    if save:
        fig = plot.get_figure()
        fig.savefig(f"{output_dir}/{mode}_{metric}_avg_bar.png")

def run_stats(
        data: pd.DataFrame,
        data_keys: List[str],
        permutations=10000) -> None:
    """Run statistical tests on the results.
    
    This function should run a series of statistical tests to determine if there is a significant
        difference between the language models for the specified metrics.
    """
    # compare all data keys to each other
    results = {}
    significant_results = {}
    for i in range(len(data_keys)):
        for j in range(i + 1, len(data_keys)):
            key_1 = data_keys[i]
            key_2 = data_keys[j]
            dist_1 = data[key_1].tolist()
            dist_2 = data[key_2].tolist()

            # Run a t-test
            t_stat, p_value = t_test(dist_1, dist_2, permutations=permutations, alternative="two-sided")
            # print(f"t-statistic: {t_stat}, p-value: {p_value}")
            results[f"{key_1} vs {key_2}"] = (t_stat, p_value)
            if p_value < 0.05:
                significant_results[f"{key_1} vs {key_2}"] = (t_stat, p_value)
                print(f"Significant difference between {key_1} and {key_2}: t-statistic = {t_stat}, p-value = {p_value}")

    return results, significant_results

if __name__ == "__main__":
    mode = "RSVP"
    path = load_experimental_data(message="Select the directory containing the simulation output")
    data = extract_sim_output(Path(path))
    final_data = process_sim_output(data)
    df = save_results(final_data, Path(path), mode=mode)

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
        plot_df = df.copy()
        plot_results(plot_df, metric, data_keys, Path(path))

        stats_df = df.copy()
        # results, sig_results = run_stats(stats_df, data_keys)
    breakpoint()