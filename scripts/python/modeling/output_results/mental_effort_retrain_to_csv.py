"""
Write the results of the mental effort model to a CSV file.

Output should be columns for partipant-run_type(matrix or rsvp) and rows for each trial window.
The input is a json file with the following structure:
# {
#     "participant_id": {
#         "run_id": {
#             "trial_window": {
#                 "auc": 0.1234,...

where run_id is the string of the run folder (ex. "CB123_RSVP_Calibration_Mon_25_Nov_2024_09hr48min50sec_-0800" or "CB123_Matrix_Calibration_Mon_25_Nov_2024_09hr48min50sec_-0800")
and trial_window is the string of the trial window (ex. "0.0:0.5").
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt



def main(filename: str):
    # load the json file
    with open(filename, "r") as f:
        data = json.load(f)
    
    # create a dataframe to store the results
    df = pd.DataFrame()
    # iterate over the participants
    for participant_id, runs in data.items():
        # iterate over the runs
        for run_id, trial_windows in runs.items():
            # iterate over the trial windows
            for trial_window, results in trial_windows.items():
                # get the auc value
                auc = results["auc"]
                # get the run type (matrix or rsvp)
                if "Matrix" in run_id:
                    run_type = "Matrix"
                elif "RSVP" in run_id:
                    run_type = "RSVP"
                else:
                    raise ValueError(f"Unknown run type in {run_id}")
                
                col_name = f"{participant_id}_{run_type}"
                # check if the column already exists
                if col_name not in df.columns:
                    # create a new column for the participant and run type
                    df[col_name] = pd.Series(dtype=float)
                # set the value for the trial window
                df.at[trial_window, col_name] = auc

                # set the index to the trial window
                df.index.name = "Trial Window"
                # set the column name to the participant and run type
                df.columns.name = "Participant-Run Type"

    # save the dataframe to a csv file
    # df.to_csv("mental_effort_results.csv")
    # get the average for each trial type (matrix or rsvp)
    df_avg = df.groupby(df.columns.str.split('_').str[1], axis=1).mean()
    ax = df_avg.plot(
        kind='bar',
        figsize=(10, 6),
        title='Average AUC for each Trial Window by Paradigm',
        xlabel='Trial Window',
        ylabel='AUC')
    
    # add the values to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    breakpoint()
    print(f"Results saved to mental_effort_results.csv")


if __name__ == "__main__":
    json_filename = 'C:\\Users\\tabme\\Desktop\\BciPy\\scripts\\python\\modeling\\output_results\\mental_effort_retrain_results_2.json'
    main(json_filename)