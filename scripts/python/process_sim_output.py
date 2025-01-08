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

SUMMARY_KEY_MAP = [
    ("N_SERIES", "total_number_series"),
    ("N_INQUIRIES", "total_inquiries"),
    ("N_SELECTIONS", "total_selections"),
    ("INQUIRIES_PER_SERIES", "inquiries_per_selection"),
    ("SELECTIONS_CORRECT", "task_summary__selections_correct"),
    ("SELECTIONS_INCORRECT", "task_summary__selections_incorrect"),
    ("SELECTIONS_CORRECT_SYMBOLS", "task_summary__selections_correct_symbols"),
    ("TYPING_ACCURACY", "task_summary__typing_accuracy"),
    ("TOTAL_SECONDS", "total_seconds")
]
SUMMARY_FILE_NAME = "summary_data.json"
LANGUAGE_MODELS = ["UNIFORM", "KENLM"] 


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
        data: List[float] = summary_data[key]
        average = sum(data) / len(data)
        std_dev = sum((x - average) ** 2 for x in data) / len(data)

        summary_stats[f"{value}_AVG"] = average
        summary_stats[f"{value}_STD"] = std_dev
    return summary_stats

def save_results(data: dict, output_dir: Path) -> None:
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
    df.to_csv(output_dir / "group_results.csv")


def process_sim_output(data: dict) -> dict:
    """Process the simulation output data to average the results for each language model across phrases.
    
    Input should be a dictionary with the following structure:

    ```python
    {
        "user1": {
            "language_model1": {
                "phrase1": {
                    "user1_phrase1_language_model1": {
                        "N_SERIES_AVG": [10, 12, 8, ...],
                        "N_SERIES_STD": [5, 6, 4, ...],
                        ...
                    },
                "phrase2": {
                    ...
                },
                "language_model2": {
                    ...
                }
            },
            "phrase2": {
                ...
            }
        },
        "user2": {
            ...
        }
    }

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
                for phrase, summary_data in phrases.items():
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

if __name__ == "__main__":
    path = load_experimental_data(message="Select the directory containing the simulation output")
    data = extract_sim_output(Path(path))
    final_data = process_sim_output(data) #WIP
    breakpoint()
    # TODO: process the data - for each language model, calculate the average and standard deviation for each language model
    save_results(final_data, Path(path))
