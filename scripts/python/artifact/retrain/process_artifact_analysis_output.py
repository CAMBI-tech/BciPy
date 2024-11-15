import csv
import pandas as pd
from bcipy.helpers.load import load_experimental_data

DATA_PATH = '/Users/scitab/Desktop/ArtifactFilteringOnlineParity1_output'
FILENAME = 'cf_nar_1_10_luckfilter_IIR_flat_all_models.csv'
PREFIX = 'cf_nar_1_10'
EXPORT_NAME = f'{PREFIX}_IIR_flat_all_models.csv'
MODELS = ['LR', 'NN', 'SVM', 'TS-LDA', 'MDM', 'PRK', 'RF']
HEADERS = [f'_{model}' for model in MODELS]

def export_flattened_csv(data_path):

    data = pd.read_csv(data_path)
    export = {}
    for metric in data:
        metric_data = data[metric]
        
        for i, participant in enumerate(metric_data):
            
            # initialize the export for this metric if it doesn't exist already
            try:
                export[i]
            except:
                export[i] = {}
            # breakpoint()
            try:
                if 'RSVP' in participant:
                    # print(participant)
                    export[i]['ID'] = participant.split("_")[0]
                else:
                    model_results = participant.split('Name')[0].split('0   ')[1:]
                    assert len(model_results) == len(MODELS), "The number of models does not equal the data processed from csv"
                    for j, (header, result) in enumerate(zip(HEADERS, model_results)):
                        parsed_result = result.split('\n')[0]

                        # catch NaN values
                        try:
                            cast_result = float(parsed_result)
                        except:
                            cast_result = 0.0
                        export[i][f'{header}_{metric}'] = cast_result
            except Exception as e:
                print(e)
                # breakpoint()
                print(f"Error processing: {export[i]['ID']}. Likely not enough samples to train a model.")
                for header in HEADERS:
                    export[i][f'{header}_{metric}'] = 0.0
                pass

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
        path = DATA_PATH

    resp = export_flattened_csv(f'{path}/{filename}')
    print("Done!")