import json
from bcipy.helpers.task import alphabet
from bcipy.helpers.load import fast_scandir
from bcipy.helpers.load import load_json_parameters
import random
import pdb
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from tabulate import tabulate

SYMBOLS = alphabet()
DECISION_THRESHOLD_RANGE = [.60, .65, .70, .75, .80]
MAX_INQUIRIES_RANGE = [3, 4, 5, 6, 7, 8, 9, 10, 11]   # max inquiries per series

def session_simulation(data, inquiry_length, max_inquiries, decision_threshold) -> list:
    series = data['series']

    _series = {
        # '0': {
        #     'time_to_series': 0,  # seconds
        # }
    } # top level key: series number, two keys: 'time_to_series'
    series_count = 0
    for inquiries in series.values():
        inq_count = 0
        series_count += 1
        for inquiry in inquiries.values():
            inq_count += 1
            _series[series_count] = {}
            likelihood = inquiry['likelihood']
            target_letter = inquiry['target_letter']

            max_idx = np.argmax(np.array(likelihood))
            max_likelihood = likelihood[max_idx]
            most_likely_letter = SYMBOLS[max_idx]
            # Calculate the end time for series
            if inq_count >= max_inquiries:
                _series[series_count]['time_to_series'] = inquiry_length * inq_count
                _series[series_count]['inq_count'] = inq_count
                if target_letter == most_likely_letter:
                    _series[series_count]['correct_selection'] = True
                else:
                    _series[series_count]['correct_selection'] = False
                break
            
            if max_likelihood >= decision_threshold:
                _series[series_count]['time_to_series'] = inquiry_length * inq_count
                _series[series_count]['inq_count'] = inq_count
                if target_letter == most_likely_letter:
                    _series[series_count]['correct_selection'] = True
                else:
                    _series[series_count]['correct_selection'] = False
                break
            
            if max_likelihood < decision_threshold:
                _series[series_count]['time_to_series'] = inquiry_length * inq_count
                _series[series_count]['inq_count'] = inq_count
                _series[series_count]['correct_selection'] = False
            
    return _series



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default=None)
    args = parser.parse_args()
    data_folder = args.data_folder 
    dataframes = []
    for _data in fast_scandir(data_folder):
        all_data = {}
        print(_data)
        # retrieve target and nontarget likelihoods from EEG for a singe sessions
        data = json.load(open(f'{_data}/session.json'))
        CP_params = load_json_parameters(f'{_data}/parameters.json')
        time_flash = float(CP_params["time_flash"]["value"])
        time_fixation = float(CP_params["time_fixation"]["value"])
        stim_length = float(CP_params["stim_length"]["value"])
        inquiry_length = time_fixation + stim_length*time_flash   # seconds
        # print(f'Inquiry length: {inquiry_length} seconds')

        # Simulate the session with different IP parameters
        for max_inquiries in MAX_INQUIRIES_RANGE:
            all_data[max_inquiries] = {}
            for decision_threshold in DECISION_THRESHOLD_RANGE:
                # print(max_inquiries, decision_threshold)
                series_info = session_simulation(data, inquiry_length, max_inquiries, decision_threshold)
                # print(series_info)

                phrase_completion_time = 0  # seconds
                correct_selection = 0
                for series_idx in range(len(series_info)):
                    phrase_completion_time = phrase_completion_time + float(series_info[series_idx+1]['time_to_series'])
                    correct_selection = correct_selection + int(series_info[series_idx+1]['correct_selection'])

                accuracy = correct_selection / len(series_info) * 100  # percentage
                accuracy = round(accuracy, 1)
                # print(f'Phrase completion time: {phrase_completion_time} seconds')
                # print(f'Accuracy: {accuracy}%')
                all_data[max_inquiries][decision_threshold] = str(phrase_completion_time)+"s, "+str(accuracy)+"%"
        # print(all_data)
        # print('\n')
        # pdb.set_trace()

        # Create a table to display the results
        df = pd.DataFrame(all_data)
        dataframes.append(df)
        print(tabulate(df, headers='keys'))
        print("\n")
   
    startrow = 0
    with pd.ExcelWriter('simulation_results.xlsx') as writer:
        for dfs in dataframes:
            dfs.to_excel(writer, sheet_name='All Subjects', startrow=startrow, header='keys')
            startrow += len(dfs.index) + 1