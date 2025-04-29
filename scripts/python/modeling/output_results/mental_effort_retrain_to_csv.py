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
    df.to_csv("mental_effort_results_2.csv")
    # get the average for each trial type (matrix or rsvp)
    df_summary = df.groupby(df.columns.str.split('_').str[1], axis=1).mean()
    df_summary_std = df.groupby(df.columns.str.split('_').str[1], axis=1).std().T.values
   
    df_summary.plot(kind='bar', yerr=df_summary_std, capsize=4, figsize=(10, 6))
    
    # ax.set_xlabel("Trial Window")
    # ax.set_ylabel("AUC")
    # ax.set_title("Mental Effort Model Results")
    # ax.legend(title="Run Type")
    plt.xticks(rotation=45)

    breakpoint()
    print(f"Results saved to mental_effort_results.csv")

    # make a new plot for RSVP and Matrix separately
    df_rsvp = df.filter(like="RSVP")
    # get the mean and std for each trial window
    df_rsvp_mean = df_rsvp.mean(axis=1)
    df_rsvp_std = df_rsvp.std(axis=1)
    
    df_rsvp_mean.plot(kind='bar', yerr=df_rsvp_std, capsize=4, figsize=(10, 6), color='lightgreen')
    plt.plot(df_rsvp_mean.index, df_rsvp_mean.values, marker='o', linestyle='-', color='red')
    max_value = df_rsvp_mean.max()
    max_index = df_rsvp_mean.idxmax()
    plt.annotate(f'Max: {max_value:.2f}', xy=(max_index, max_value), xytext=(max_index, max_value + 0.05), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10, ha='center')
    plt.title("RSVP Mental Effort Model Results")

    # Now for the Matrix
    df_matrix = df.filter(like="Matrix")
    df_matrix_mean = df_matrix.mean(axis=1)
    df_matrix_std = df_matrix.std(axis=1)

    df_matrix_mean.plot(kind='bar', yerr=df_matrix_std, capsize=4, figsize=(10, 6), color='lightblue')
    plt.plot(df_matrix_mean.index, df_matrix_mean.values, marker='o', linestyle='-', color='blue')
    max_value = df_matrix_mean.max()
    max_index = df_matrix_mean.idxmax()
    plt.annotate(f'Max: {max_value:.2f}', xy=(max_index, max_value), xytext=(max_index, max_value + 0.05), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10, ha='center')
    plt.title("Matrix Mental Effort Model Results")
    plt.xlabel("Trial Window")

    
    breakpoint()    





if __name__ == "__main__":
    json_filename = 'C:\\Users\\tabme\\Desktop\\BciPy\\scripts\\python\\modeling\\output_results\\ME_trial_window_data.json'
    main(json_filename)