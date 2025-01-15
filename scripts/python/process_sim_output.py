"""Process Sim Output.

This script processes the output of the simulation and generates a summary of the results. The current output directory structure is as follows:

```text
output_dir/
    user1/
        language_model1/
            phrase1/
                user1_phrase1_language_model1_SIM_datetime/
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
        language_model2/
            phrase1/
                user1_phrase1_language_model2_SIM_datetime/
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

from matplotlib import pyplot as plt

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
LANGUAGE_MODELS = ["UNIFORM", "KENLM", "CAUSAL"] 


def extract_sim_output(output_dir: Path) -> None:
    """Process the simulation output and generate a summary of the results."""
    
    output_data = {}
    for user in output_dir.iterdir():
        if user.is_dir():
            output_data[user.name] = {}
            for language_model in user.iterdir():
                if language_model.is_dir():
                    output_data[user.name][language_model.name] = {}
                    for phrase in language_model.iterdir():
                        if phrase.is_dir():
                            output_data[user.name][language_model.name][phrase.name] = {}
                            for sim_run in phrase.iterdir():
                                if sim_run.is_dir():
                                    summary_data_path = sim_run / SUMMARY_FILE_NAME
                                    summary_data = summary_stats_for_sim_output(summary_data_path)
                                    output_data[user.name][language_model.name][phrase.name]= summary_data
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

def save_results(data: dict, output_dir: Path) -> pd.DataFrame:
    """Save the results to the output directory as a CSV file.
    
    Data should be a dictionary with the following structure:

    ```python
    {
        "user1": {
            "language_model1": {
                "phrase1": {
                    "N_SERIES_AVG": 10,
                    "N_SERIES_STD": 2,
                    "N_INQUIRIES_AVG": 5,
                    "N_INQUIRIES_STD": 1,
                    ...
                },
                "phrase2": {
                    ...
                }
            },
            "language_model2": {
                ...
            }
        },
        "user2": {
            ...
        }
    }
    """
    df = pd.DataFrame(data)
    df = df.transpose()
    df.to_csv(output_dir / RESULTS_FILE_NAME)
    return df


def process_sim_output(data: dict) -> dict:
    """Process the simulation output data to average the results for each language model across phrases.

    Output should be a dictionary with the following structure:

    {
        "user1": {
            "language_model1": {
                "N_SERIES_AVG": 10,
                "N_SERIES_STD": 2,
                "N_INQUIRIES_AVG": 5,
                "N_INQUIRIES_STD": 1,
                ...
            },
            "language_model2": {
                ...
            }
        },
        "user2": {
            ...
    }
    """
    # Loop over the phrases for each language model and calculate the average and standard deviation for each metric
    final_data = {}
    for user, language_model in data.items():
        final_data[user] = {}
        for language_model, phrases in language_model.items():
            if language_model in LANGUAGE_MODELS:
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
                    final_data[user][f'{language_model}_{key}'] = average
                    final_data[user][f'{language_model}_{key}_STD']  = std_dev
            else:
                print(f"Skipping language model {language_model} as it is not in the list of valid language models")
    return final_data

def grab_data_keys(metrics: List[str], models: List[str]) -> List[str]:
    """Generate the data keys for the specified metrics and models."""
    data_keys = []
    for model in models:
        for metric in metrics:
            key = f"{model}_{metric}"
            data_keys.append(key)
    return data_keys

def plot_results(
        data: pd.DataFrame,
        metric: str,
        data_keys: List[str],
        output_dir: Path,
        show=True,
        save=True) -> None:
    """Plot the results of the simulation."""
    # BAR PLOT
    plot = data[data_keys].plot(kind="bar", figsize=(10, 6))
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.xlabel("User")
    plt.title(f"{metric} by Language Model")

    if show:
        plt.show()
    
    if save:
        fig = plot.get_figure()
        fig.savefig(f"{output_dir}/{metric}_bar.png")

    colors = dict(boxes="#c5cbca", whiskers="#0f0f0f", medians="#046565", caps="#0f0f0f")
    # BOXPLOT
    boxplot = data.boxplot(column=data_keys, figsize=(10, 6), grid=False, fontsize=15, color=colors, patch_artist=True, notch=True)
    plt.ylabel(metric)
    # The x-axis is the language model 
    label_text = [key.split("_")[0] for key in data_keys]
    plt.xticks(range(1, len(data_keys) + 1), label_text)
    # plt.title(f"{metric} by Language Model")

    if show:
        plt.show()

    if save:
        fig = boxplot.get_figure()
        fig.savefig(f"{output_dir}/{metric}_box.png")
    


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
    for i in range(len(data_keys)):
        for j in range(i + 1, len(data_keys)):
            key_1 = data_keys[i]
            key_2 = data_keys[j]
            print(f"Comparing {key_1} and {key_2}")
            dist_1 = data[key_1].tolist()
            dist_2 = data[key_2].tolist()

            # Run a t-test
            t_stat, p_value = t_test(dist_1, dist_2, permutations=permutations, alternative="two-sided")
            print(f"t-statistic: {t_stat}, p-value: {p_value}")
            results[f"{key_1} vs {key_2}"] = (t_stat, p_value)
    return results

if __name__ == "__main__":
    # TODO VERIFY CHAR /MIN CALCULATION AND STATS
    path = load_experimental_data(message="Select the directory containing the simulation output")
    data = extract_sim_output(Path(path))
    final_data = process_sim_output(data)
    df = save_results(final_data, Path(path))

    # for plotting and statistical analysis
    processing_metrics = [
        "TOTAL_SECONDS_AVG",
        "TOTAL_SECONDS_STD_STD",
        "CHAR_PER_MIN_AVG",
        "CHAR_PER_MIN_STD_STD",
        "N_INQUIRIES_AVG",
        "N_INQUIRIES_STD_STD",
        "TYPING_ACCURACY_AVG",
        "TYPING_ACCURACY_STD_STD",
        "SELECTIONS_INCORRECT_AVG",
        "SELECTIONS_INCORRECT_STD_STD",
        ]
    # processing_metrics = ["TYPING_ACCURACY_AVG"]
    # processing_metrics = ["N_SERIES_AVG"]
    processing_models = LANGUAGE_MODELS

    for metric in processing_metrics:
        data_keys = grab_data_keys([metric], processing_models)
        plot_df = df.copy()
        plot_results(plot_df, metric, data_keys, Path(path))

        stats_df = df.copy()
        results = run_stats(stats_df, data_keys)
    breakpoint()
