"""
Compare Model Output 
--------------------

This script takes in a json file with the following format:

{
    "user_id": { # note there are 25 users
        "of: {
            "model": {
                "inquiry_number": {
                    "eeg_likelihood_evidence": [
                        float, float, float, ... (length = 26)
                    ],
                    "target_idx": Optional[int],
                    "nontarget_idx": int
                }
                }
            },
        "cf": {
            "model": {
                "inquiry_number": {
                    "eeg_likelihood_evidence": [
                        float, float, float, ... (length = 26)
                    ],
                    "target_idx": Optional[int],
                    "nontarget_idx": int
                }
                }
            },
        }
    
    }
}

The script will compare the output of the OF and CF models for each inquiry, distinquishing target and nontarget letters.

"""

import json
import numpy as np
from bcipy.gui.file_dialog import ask_filename
import matplotlib.pyplot as plt

def load_json_file(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def extract_and_score_model_output(data: dict) -> dict:
    results = {}
    for user_id, user_data in data.items():
        results[user_id] = {}
        for model_type, model_data in user_data.items():
            # This will either be 'of' or 'cf'
            results[user_id][model_type] = {}

            # Loop over model data and score whether the model correctly identified the target letter
            # Where a value of > 1.0 for a target letter indicates the model correctly identified the target
            # letter. A value of <= 1.0 indicates the model incorrectly identified the target letter.
            # Conversely, a value of > 1.0 for a nontarget letter indicates the model incorrectly identified the letter.

            for inquiry_number, inquiry_data in model_data['model'].items():

                # Initialize the results dictionary
                results[user_id][model_type][inquiry_number] = {}
                results[user_id][model_type][inquiry_number]['nontarget'] = []
                results[user_id][model_type][inquiry_number]['target'] = None

                # Get the model output
                eeg_likelihood_evidence = inquiry_data['eeg_likelihood_evidence']
                target_idx = inquiry_data['target_idx']
                nontarget_idx = inquiry_data['nontarget_idx']
                if target_idx:
                    value = eeg_likelihood_evidence[target_idx]
                    results[user_id][model_type][inquiry_number]['target'] = value
            
                for idx in nontarget_idx:
                    value = eeg_likelihood_evidence[idx]
                    results[user_id][model_type][inquiry_number]['nontarget'].append(value)


            # loop over the results and score the model output
            for inquiry_number, inquiry_data in model_data['model'].items():
                target_idx = inquiry_data['target_idx']
                
                target_value = results[user_id][model_type][inquiry_number]['target']
                if target_value is None:
                    results[user_id][model_type][inquiry_number]['target_score'] = 'NA'
                elif target_value > 1.0:
                    results[user_id][model_type][inquiry_number]['target_score'] = 'TP'
                else:
                    results[user_id][model_type][inquiry_number]['target_score'] = 'FN'
                
                nontarget_values = results[user_id][model_type][inquiry_number]['nontarget']
                results[user_id][model_type][inquiry_number]['nontarget_average'] = np.mean(nontarget_values)
                results[user_id][model_type][inquiry_number]['nontarget_score'] = []
                for nontarget_value in nontarget_values:
                    if nontarget_value > 1.0:
                        results[user_id][model_type][inquiry_number]['nontarget_score'].append('FP')
                    else:
                        results[user_id][model_type][inquiry_number]['nontarget_score'].append('TN')
                    

    return results


def evaluate_model_output(score_results: dict) -> dict:
    """
    Evaluate the model output using the FP, FN, TP, TN scores.

    FP = False Positive
    FN = False Negative
    TP = True Positive
    TN = True Negative

    The scores are then tallied, and several scores (acc, ba, mccc) are calculated.

    ACC = (TP + TN) / (TP + TN + FP + FN)
    BA = 0.5 * (TP / (TP + FN) + TN / (TN + FP))
    MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    -------------

    Results input is in the following format:
    
        {
            "user_id": {
                "of": {
                    "inquiry_number": {
                        "nontarget": [float, float, float, ...],
                        "target": float,
                        "target_score": str,
                        "nontarget_average": float,
                        "nontarget_score": [str, str, str, ...]
                    },
                "cf": {
                    "inquiry_number": {
                        "nontarget": [float, float, float, ...],
                        "target": float,
                        "target_score": str,
                        "nontarget_average": float,
                        "nontarget_score": [str, str, str, ...]
                    },
                }
            }

    -------------------------

    Output should be as follows:

        {
            "user_id": {
                "of": {
                    "TP": int,
                    "FP": int,
                    "TN": int,
                    "FN": int,
                    "ACC": float,
                    "BA": float,
                    "MCC": float
                },
                "cf": {
                    "TP": int,
                    "FP": int,
                    "TN": int,
                    "FN": int,
                    "ACC": float,
                    "BA": float,
                    "MCC": float
                }
            }
        }
    """
    results = {}
    for user_id, user_data in score_results.items():
        results[user_id] = {}
        for model_type, model_data in user_data.items():
            results[user_id][model_type] = {}
            results[user_id][model_type]['TP'] = 0
            results[user_id][model_type]['FP'] = 0
            results[user_id][model_type]['TN'] = 0
            results[user_id][model_type]['FN'] = 0

            print(f"User: {user_id}, Model: {model_type} - Scoring...")
            print("-------------------------------------------------")
            # print(f"{results}")

            for inquiry_number, inquiry_data in model_data.items():
                print(f"{user_id}-{inquiry_number}-{inquiry_data}")
                target_score = inquiry_data['target_score']
                nontarget_scores = inquiry_data['nontarget_score']

                if target_score == 'NA':
                    continue
                elif target_score == 'TP':
                    results[user_id][model_type]['TP'] += 1
                else:
                    results[user_id][model_type]['FN'] += 1

                for score in nontarget_scores:
                    if score == 'FP':
                        results[user_id][model_type]['FP'] += 1
                    else:
                        results[user_id][model_type]['TN'] += 1

            # Calculate the scores
            TP = results[user_id][model_type]['TP']
            FP = results[user_id][model_type]['FP']
            TN = results[user_id][model_type]['TN']
            FN = results[user_id][model_type]['FN']

            results[user_id][model_type]['ACC'] = (TP + TN) / (TP + TN + FP + FN)
            results[user_id][model_type]['BA'] = 0.5 * (TP / (TP + FN) + TN / (TN + FP))
            results[user_id][model_type]['MCC'] = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return results

def save_results(results: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(results, f)


def plot_average_evaluation_results(eval_results: dict) -> None:
    """
    Plot the evaluation results.

    eval_results:
    {
            "user_id": {
                "of": {
                    "TP": int,
                    "FP": int,
                    "TN": int,
                    "FN": int,
                    "ACC": float,
                    "BA": float,
                    "MCC": float
                },
                "cf": {
                    "TP": int,
                    "FP": int,
                    "TN": int,
                    "FN": int,
                    "ACC": float,
                    "BA": float,
                    "MCC": float
                }
            }
        }
    """
    # get an average of BA and MCC
    ba_values_of = []
    mcc_values_of = []
    ba_values_cf = []
    mcc_values_cf = []
    for user_id, user_data in eval_results.items():
        ba_values_of.append(user_data['of']['BA'])
        mcc_values_of.append(user_data['of']['MCC'])
        ba_values_cf.append(user_data['cf']['BA'])
        mcc_values_cf.append(user_data['cf']['MCC'])
    
    avg_ba_of = np.mean(ba_values_of)
    avg_mcc_of = np.mean(mcc_values_of)
    avg_ba_cf = np.mean(ba_values_cf)
    avg_mcc_cf = np.mean(mcc_values_cf)

    print(f"Average BA OF: {avg_ba_of}")
    print(f"Average MCC OF: {avg_mcc_of}")
    print(f"Average BA CF: {avg_ba_cf}")
    print(f"Average MCC CF: {avg_mcc_cf}")

    # plot the results with a bar chart (0-1) and add labels with the values
    fig, ax = plt.subplots()
    x = np.arange(4)
    width = 0.35
    ba_values = [avg_ba_of, avg_ba_cf, avg_mcc_of, avg_mcc_cf]
    labels = ['BA OF', 'BA CF', 'MCC OF', 'MCC CF']
    ax.bar(x, ba_values, width, label='Scores')
    ax.set_ylabel('Scores')
    ax.set_title('Model Evaluation Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # Add the values to the bars
    for i, v in enumerate(ba_values):
        ax.text(i, v + 0.01, str(round(v, 2)), color='black', ha='center')

    # make the colors different for ba and mcc
    ax.bar(x[2:], ba_values[2:], width, color='r')
    ax.bar(x[:2], ba_values[:2], width  , color='b')

    plt.show()

def plot_scatter_evaluation_results(eval_results):
    """
    Plot the evaluation results in a scatter plot.

    eval_results:
    {
            "user_id": {
                "of": {
                    "TP": int,
                    "FP": int,
                    "TN": int,
                    "FN": int,
                    "ACC": float,
                    "BA": float,
                    "MCC": float
                },
                "cf": {
                    "TP": int,
                    "FP": int,
                    "TN": int,
                    "FN": int,
                    "ACC": float,
                    "BA": float,
                    "MCC": float
                }
            }
        }
    """
    # get the values for the scatter plot
    ba_values_of = []
    mcc_values_of = []
    ba_values_cf = []
    mcc_values_cf = []
    for user_id, user_data in eval_results.items():
        ba_values_of.append(user_data['of']['BA'])
        mcc_values_of.append(user_data['of']['MCC'])
        ba_values_cf.append(user_data['cf']['BA'])
        mcc_values_cf.append(user_data['cf']['MCC'])
    
    # plot a box plot for the BA
    fig, ax = plt.subplots()
    x = np.arange(2)
    ba_values = [ba_values_of, ba_values_cf]
    labels = ['BA OF', 'BA CF']
    ax.boxplot(ba_values)
    ax.set_title('BA Model Evaluation Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.show()

    # plot a box plot for the MCC
    fig, ax = plt.subplots()
    x = np.arange(2)
    mcc_values = [mcc_values_of, mcc_values_cf]
    labels = ['MCC OF', 'MCC CF']
    ax.boxplot(mcc_values)
    ax.set_title('MCC Model Evaluation Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.show()

    # run a t-test to see if the scores are significantly different
    from scipy.stats import ttest_ind
    t, p = ttest_ind(ba_values_of, ba_values_cf)
    print(f"BA T-Test: {t}, {p}")
    t, p = ttest_ind(mcc_values_of, mcc_values_cf)
    print(f"MCC T-Test: {t}, {p}")





if __name__ in "__main__":
    path = ask_filename('*json')
    data = load_json_file(path)
    results = extract_and_score_model_output(data)
    save_results(results, 'score_results.json')
    eval_results = evaluate_model_output(results)
    save_results(eval_results, 'eval_results.json')
    plot_average_evaluation_results(eval_results)
    plot_scatter_evaluation_results(eval_results)
    breakpoint()