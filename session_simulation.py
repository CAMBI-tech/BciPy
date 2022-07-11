
import json
from bcipy.helpers.task import alphabet
from bcipy.helpers.load import fast_scandir
import random
import pdb
import numpy as np

SYMBOLS = alphabet()
DECISION_THRESHOLD = 0.8 # .60, .65 , .70, .75, .80, .85, .90, .95
MAX_INQUIRIES = 11 # 3-11
INQUIRY_LENTH = 3

def session_simulation(data) -> list:
    series = data['series']

    _series = {
        # '0': {
        #     'time_to_series': 0,
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
            if inq_count >= MAX_INQUIRIES:
                _series[series_count]['time_to_series'] = INQUIRY_LENTH * inq_count
                _series[series_count]['inq_count'] = inq_count
                if target_letter == most_likely_letter:
                    _series[series_count]['correct_selection'] = True
                else:
                    _series[series_count]['correct_selection'] = False
                break
            
            if max_likelihood >= DECISION_THRESHOLD:
                _series[series_count]['time_to_series'] = INQUIRY_LENTH * inq_count
                _series[series_count]['inq_count'] = inq_count
                if target_letter == most_likely_letter:
                    _series[series_count]['correct_selection'] = True
                else:
                    _series[series_count]['correct_selection'] = False
                break
            
    return _series



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default=None)
    args = parser.parse_args()
    data_folder = args.data_folder
    all_data = []
    for _data in fast_scandir(data_folder):
        print(_data)
        # retrieve target and nontarget likelihoods from EEG for a singe sessions
        data = json.load(open(f'{_data}/session.json'))
        series_info = session_simulation(data)
        print(series_info)
        all_data.append(series_info)
    
    pdb.set_trace()






