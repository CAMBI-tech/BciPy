"""SCRD DV Script.

This script is used to generate the SCRD DV report. It will output the summary measures across a single
participant's copy phrase data.

The data is expected to be in the following format:

user_id/ * this is what the user should load from the file dialog
    /date
        /experiment
            /date_timestamp
                /*copy_phrase/
                    raw_data.csv
                    triggers.txt
                    session.json
                /*copy_phrase/

We want the following averaged measures:

Accuracy
Copy_Rate
Correct_Rate
Inquiries_per_selection
Inquiry_Total
Selections_Correct
Selections_Correct_Backspace
Selections_Correct_Letters
Selections_Incorrect
Selections_Total
Session_Length
Session_Minutes
IP_Condition
Switch_per_selection
Switch_Total
Switch_Response_Time

Missing measures will be written as "N/A".

We get the following measures from the copy phrase session.json:

  "total_time_spent": 525.39,
  "total_minutes": 8.76,
  "total_number_series": 10,
  "total_inquiries": 64,
  "total_selections": 10,
  "inquiries_per_selection": 6.4,
  "task_summary": {
    "selections_correct": 3,
    "selections_incorrect": 7,
    "selections_correct_symbols": 1,
    "switch_total": 0,
    "switch_per_selection": 0.0,
    "switch_response_time": null,
    "typing_accuracy": 0.3,
    "correct_rate": 0.3426036152030171,
    "copy_rate": 0.11420120506767235
"""
import os
import json
import glob
import pandas as pd

from bcipy.config import SESSION_DATA_FILENAME, DEFAULT_PARAMETERS_FILENAME
from bcipy.helpers.load import load_json_parameters

DEFAULT_EXPERIMENT = "SCRD_Control"
MEASURES = [
    "User_ID",
    "Experiment",
    "Date",
    "Date_Time",
    "Accuracy",
    "Copy_Rate",
    "Correct_Rate",
    "Inquiries_Per_Selection",
    "Inquiry_Total",
    "Selections_Correct",
    "Selections_Correct_Letters",
    "Selections_Incorrect",
    "Selections_Total",
    "Session_Length",
    "Session_Minutes",
    "Switch_per_selection",
    "Switch_Total",
    "Switch_Response_Time",
    "IP_Condition"
]
MISSING_MEASURE = "N/A"

def load_session_data(path: str) -> dict:
    """Load the session data from the given path.

    Args:
        path (str): The path to the session data.

    Returns:
        pd.DataFrame: The session data.
    """
    # load the json files
    path = os.path.join(path, SESSION_DATA_FILENAME)
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_summary_measures(measures: dict, output: str):
    """Write the summary measures to the output directory.

    Args:
        measures (dict): The summary measures.
        output (str): The path to the output directory.
    """
    csv_output = os.path.join(output, "dv_summary.csv")
    # write csv but don't write the row if date time already exists
    if not os.path.exists(csv_output):
        df = pd.DataFrame(measures, index=[0])
        df.to_csv(csv_output, mode='a', header=True, index=False)
    else:
        df = pd.read_csv(csv_output)
        # If the date time already exists, don't write the row
        if measures['Date_Time'] in df['Date_Time'].values:
            return
        df = pd.DataFrame(measures, index=[0])
        df.to_csv(csv_output, mode='a', header=False, index=False)

def calculate_summary_measures(data: dict, experiment_id: str) -> dict:
    # Calculate the summary measures using an average across all the sessions
    measures = {}
    # intialize the measures
    for measure_id in MEASURES:
        measures[measure_id] = []

    for date_time in data[experiment_id]:
        session_data = data[experiment_id][date_time]

        # iterate over the sessions and cast the measures to the correct type
        for session in session_data.values():
            measures['Accuracy'].append(float(session['task_summary']['typing_accuracy']))
            measures['Copy_Rate'].append(float(session['task_summary']['copy_rate']))
            measures['Correct_Rate'].append(float(session['task_summary']['correct_rate']))
            measures['Inquiries_Per_Selection'].append(int(session['inquiries_per_selection']))
            measures['Inquiry_Total'].append(int(session['total_inquiries']))
            measures['Selections_Total'].append(int(session['total_selections']))
            measures['Selections_Correct'].append(int(session['task_summary']['selections_correct']))
            measures['Selections_Incorrect'].append(int(session['task_summary']['selections_incorrect']))
            measures['Selections_Correct_Letters'].append(int(session['task_summary']['selections_correct_symbols']))
            measures['Session_Length'].append(float(session['total_time_spent']))
            measures['Session_Minutes'].append(float(session['total_minutes']))
            switch_response_time = session['task_summary']['switch_response_time']
            if switch_response_time:
                measures['Switch_per_selection'].append(float(session['task_summary']['switch_per_selection']))
                measures['Switch_Total'].append(int(session['task_summary']['switch_total']))
                measures['Switch_Response_Time'].append(switch_response_time)
            else:
                measures['Switch_per_selection'].append(MISSING_MEASURE)
                measures['Switch_Total'].append(MISSING_MEASURE)
                measures['Switch_Response_Time'].append(MISSING_MEASURE)
            

    # calculate the average of the measures
    for measure_id in MEASURES:
        try:
            measures[measure_id] = sum(measures[measure_id]) / len(measures[measure_id])
        except ZeroDivisionError:
            measures[measure_id] = MISSING_MEASURE
        except TypeError:
            measures[measure_id] = MISSING_MEASURE
        except ValueError:
            measures[measure_id] = MISSING_MEASURE
    return measures

def calculate_correct_backspace(data: dict) -> int:
    pass

def iterate_experiment_data(user_data_path: str, experiment: str, output: str):
    """Iterate over the experiment data.

    Args:
        path (str): The path to the experiment data.
        experiment (str): The experiment to analyze.
        output (str): The path to the output directory.

    """
    # grab the user_id from the path
    user_id = os.path.basename(user_data_path)
    session_data = {}
    measures_all = {}
    # iterate over the user directories
    for date in os.listdir(user_data_path):
        if not os.path.isdir(os.path.join(user_data_path, date)):
            continue
        # iterate over the experiment directories
        session_data = {}
        full_date_path = os.path.join(user_data_path, date)
        for experiment_id in os.listdir(full_date_path):
            full_experiment_path = os.path.join(full_date_path, experiment_id)
            if experiment != experiment_id or not os.path.isdir(full_experiment_path):
                continue
            else:
                session_data[experiment_id] = {}
    
                for date_time in os.listdir(full_experiment_path):
                    full_date_time_path = os.path.join(user_data_path, date, experiment_id, date_time)
                    if not os.path.isdir(full_date_time_path):
                        continue
                    
                    session_data[experiment_id][date_time] = {}
                    # look for the copy phrase data using the glob pattern *Copy_Phrase*
                    copy_phrase_paths = glob.glob(os.path.join(full_date_time_path, "*Copy_Phrase*"))
                    if not copy_phrase_paths:
                        print(f"No copy phrase data found for {user_id} on {date_time}")
                        continue
                    # load the session data
                    i = 0
                    for copy_phrase_data in copy_phrase_paths:
                        session_data[experiment_id][date_time][i] = {}
                        data = load_session_data(copy_phrase_data)
                        session_data[experiment_id][date_time][i] = data
                        i += 1

                    # load a single parameters file (we assume they are all the same)
                    parameters = load_json_parameters(
                        f"{copy_phrase_data}/{DEFAULT_PARAMETERS_FILENAME}",
                        value_cast=True)
                    # calculate the summary measures
                    measures = calculate_summary_measures(session_data, experiment_id)
                    measures['User_ID'] = user_id
                    measures['Date'] = date
                    measures['Date_Time'] = date_time
                    measures['Experiment'] = experiment_id
                    if parameters['show_preview_inquiry']:
                        preview_inquiry_progress_method = parameters['preview_inquiry_progress_method']
                        if preview_inquiry_progress_method == 0:
                            measures['IP_Condition'] = "IPO"
                        elif preview_inquiry_progress_method == 1:
                            measures['IP_Condition'] = "PTC"
                        elif preview_inquiry_progress_method == 2:
                            measures['IP_Condition'] = "PTS"
                    else:
                        measures['IP_Condition'] = MISSING_MEASURE

                    # write the summary measures
                    write_summary_measures(measures, output)
                    

if __name__ in "__main__":
    import argparse
    from bcipy.gui.file_dialog import ask_directory

    parser = argparse.ArgumentParser(description="SCRD DV Script")
    parser.add_argument("--path", type=str, help="The path to the user data directory", required=False, default=None)
    parser.add_argument("--output", type=str, help="The path to the output directory", required=False, default=None)
    parser.add_argument(
        "--experiment",
        type=str,
        help="The experiment to analyze copy phrase data from",
        required=False,
        default=DEFAULT_EXPERIMENT)

    # Get the directory containing the data
    args = parser.parse_args()
    if not args.path:
        path = ask_directory("Select the directory containing the copy phrase data")
    else:
        path = args.path

    # Get the output directory
    if not args.output:
        output = path
    else:
        # Check if the output directory exists
        output = args.output

    if not os.path.exists(output):
        raise ValueError(f"Output directory does not exist: {output}")
    
    # iterate over the experiment data
    iterate_experiment_data(path, args.experiment, output)
    

    
