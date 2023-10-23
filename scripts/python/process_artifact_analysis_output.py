import csv
import pandas as pd
from bcipy.helpers.load import load_experimental_data

FILENAME = 'WD_NAR_IIR_all_models.csv'
PREFIX = 'WD_NAR_IIR'
EXPORT_NAME = f'{PREFIX}_flat_all_models.csv'
MODELS = ['LR', 'NN', 'SVM', 'LDA', 'PRK', 'RF']
HEADERS = [f'WD_NAR_IIR_{model}' for model in MODELS]

def export_flattened_csv(data_path):

    data = pd.read_csv(data_path)
    export = {}
    for metric in data:
        metric_data = data[metric]
        
        for i, participant in enumerate(metric_data):
            
            try:
                export[i]
            except:
                export[i] = {}

            if 'RSVP' in participant:
                # print(participant)
                export[i]['ID'] = participant.split("_")[0]
            else:
                model_results = participant.split('Name')[0].split('0   ')[1:]
                assert len(model_results) == len(MODELS), "The number of models does not equal the data processed from csv"
                for j, (header, result) in enumerate(zip(HEADERS, model_results)):
                    export[i][f'{header}_{metric}'] = float(result.split('\n')[0])

    new_data = pd.DataFrame.from_dict(export).transpose()
    new_data.to_csv(EXPORT_NAME)

    return export, new_data


if __name__ == '__main__':
    import argparse
    # let the user chose the data folder
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the directory with raw_data to be converted',
        required=False)

    parser.add_argument('-f', '--filename', default=FILENAME)
    args = parser.parse_args()

    path = args.path
    filename = args.filename
    if not path:
        path = load_experimental_data()

    resp = export_flattened_csv(f'{path}/{filename}')
    breakpoint()