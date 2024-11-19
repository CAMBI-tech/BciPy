"""
Compare Model Output 
--------------------

This script takes in a json file with the following format:

## Paired T-test results
# BA T-Test: 1.6688799601136604, 0.10813634332002026
# MCC T-Test: 1.038312827493233, 0.30947283622952343

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
import seaborn as sns


class ConfusionMatrix:
    """Confusion Matrix.

    Confusion matrix for evaluating model outputs.
    
    TP = True Positive
    FP = False Positive
    TN = True Negative
    FN = False Negative
    """

    def __init__(self, TP: int, FP: int, TN: int, FN: int):
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN

    def __repr__(self):
        return f"TP: {self.TP}, FP: {self.FP}, TN: {self.TN}, FN: {self.FN}"
    
    @property
    def count(self) -> int:
        return self.TP + self.FP + self.TN + self.FN

    @property
    def accuracy(self) -> float:
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
    
    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.TP / (self.TP + self.FN) + self.TN / (self.TN + self.FP))
    
    @property
    def mcc(self) -> float:
        return (
            (self.TP * self.TN - self.FP * self.FN)
             / np.sqrt((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN))
            )


def load_json_file(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def extract_and_score_model_output(data: dict, trials_per_inquiry: int = 10) -> dict:
    results = {}
    for user_id, user_data in data.items():
        results[user_id] = {}
        for model_type, model_data in user_data.items():
            # This will either be 'of' or 'cf'
            results[user_id][model_type] = {}
            results[user_id][model_type]['selections'] = model_data['selections']

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
                if target_idx is not None:
                    value = eeg_likelihood_evidence[target_idx]
                    results[user_id][model_type][inquiry_number]['target'] = value

                if target_idx is None:
                    # If there are no target letters, then there should be trials_per_inquiry nontarget letters
                    if not len(nontarget_idx) == trials_per_inquiry:
                        raise ValueError(f"Expected {trials_per_inquiry} nontarget letters. {len(nontarget_idx)} were found.")
                else:
                    # If there are target letters, then there should be trials_per_inquiry - 1 nontarget letters
                    if not len(nontarget_idx) == trials_per_inquiry - 1:
                        raise ValueError(f"Expected {trials_per_inquiry - 1} nontarget letters. {len(nontarget_idx)} were found.")
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


def evaluate_model_output(score_results: dict, trials_per_inquiry: int = 10) -> dict:
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

            inquiry_counter = 0
            selection_estimate = model_data['selections']
            # pop the selections from the model data
            model_data.pop('selections')
            for inquiry_number, inquiry_data in model_data.items():
                inquiry_counter += 1
                print(f"{user_id}-{inquiry_number}-{inquiry_data}")
                target_score = inquiry_data['target_score']
                nontarget_scores = inquiry_data['nontarget_score']

                all_non_target_scores = False
                if target_score == 'NA':
                    print('NA')
                    all_non_target_scores = True
                elif target_score == 'TP':
                    results[user_id][model_type]['TP'] += 1
                else:
                    results[user_id][model_type]['FN'] += 1

                if all_non_target_scores:
                    assert len(nontarget_scores) == 10, f"There should be 10 nontarget scores. {len(nontarget_scores)} were found."
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
            conf_matrix = ConfusionMatrix(TP, FP, TN, FN)

            # validate the count of the confusion matrix and the number of inquiries match
            assert conf_matrix.count == (inquiry_counter * trials_per_inquiry), \
                ("The count of the confusion matrix should equal the number of inquiries. "
                 f"{conf_matrix.count} != {inquiry_counter}")
            
            selection_estimate = selection_estimate
            assert selection_estimate <= inquiry_counter, (
                "The selection estimate should be less than or equal to the number of inquiries.")
            inquiries_per_selection = inquiry_counter / selection_estimate

            # Calculate the scores
            results[user_id][model_type]['ACC'] = conf_matrix.accuracy
            results[user_id][model_type]['BA'] = conf_matrix.balanced_accuracy
            results[user_id][model_type]['MCC'] = conf_matrix.mcc

            # Calculate estimated ITR
            results[user_id][model_type]['ITR'] = calculate_itr(
                conf_matrix,
                trials_per_inquiry=10,
                flash_time=0.2,
                inquiries_per_selection=inquiries_per_selection
            )

    return results


def calculate_itr(
        conf_matrix: ConfusionMatrix,
        trials_per_inquiry: int = 10,
        flash_time: float = 0.2,
        inquiries_per_selection: float = 1.0) -> None:
    """
    conf_matrix: ConfusionMatrix = ConfusionMatrix(TP, FP, TN, FN)
    trials_per_inquiry: int = 10 * referred to as stim_number in BciPy
    flash_time: float = 0.2 * referred to as time_flash in BciPy
    buffer: float = 1.0 * referred to as task_buffer_length in BciPy is the time in seconds between trials
    """
    # Calculate ITR (bit per second)
    # ITR = B * Q
    # B = log2(N) + P * log2(P) + (1 - P) * log2((1 - P) / (N - 1))
    # Q = S / T
    # N = number of targets
    # P = accuracy
    # S = number of inquiries
    # N = symbol set
    # T = total time in minutes as estimated by the formula below (T_min)
    N = conf_matrix.TP + conf_matrix.FN
    trial_count = N + conf_matrix.FP + conf_matrix.TN
    isi_time = 0.1
    fixation_time = 0.5
    T_inquiry = (trials_per_inquiry * flash_time) + (trials_per_inquiry * isi_time) + fixation_time # seconds per inquiry
    N_inquiry = trial_count / trials_per_inquiry # number of inquiries
    T_sec = (N_inquiry * T_inquiry) * inquiries_per_selection # total time in seconds
    T_min = (T_sec / 60) # total time in minutes
    Q = N_inquiry / T_sec
    P = conf_matrix.accuracy
    B = np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))
    ITR = B * Q
    print(f"ITR: {ITR}")
    return ITR

def save_results(results: dict, path: str) -> None:
    """Save the results to a json file."""
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
    plot_confusion_matrix(eval_results)
    plot_ba_mcc(eval_results)


def plot_confusion_matrix(eval_results: dict) -> None:
    """
    Plot the confusion matrix of the evaluation results.
    """
    TP_values_of = []
    FP_values_of = []
    TN_values_of = []
    FN_values_of = []

    TP_values_cf = []
    FP_values_cf = []
    TN_values_cf = []
    FN_values_cf = []

    for user_id, user_data in eval_results.items():
        TP_values_of.append(user_data['of']['TP'])
        FP_values_of.append(user_data['of']['FP'])
        TN_values_of.append(user_data['of']['TN'])
        FN_values_of.append(user_data['of']['FN'])

        TP_values_cf.append(user_data['cf']['TP'])
        FP_values_cf.append(user_data['cf']['FP'])
        TN_values_cf.append(user_data['cf']['TN'])
        FN_values_cf.append(user_data['cf']['FN'])

    # get the average values
    avg_TP_of = np.mean(TP_values_of)
    avg_FP_of = np.mean(FP_values_of)
    avg_TN_of = np.mean(TN_values_of)
    avg_FN_of = np.mean(FN_values_of)

    avg_TP_cf = np.mean(TP_values_cf)
    avg_FP_cf = np.mean(FP_values_cf)
    avg_TN_cf = np.mean(TN_values_cf)
    avg_FN_cf = np.mean(FN_values_cf)

    # Make two plots for OF and CF
    fig, ax = plt.subplots()
    x = np.arange(4)
    width = 0.4
    values_of = [avg_TP_of, avg_FP_of, avg_TN_of, avg_FN_of]
    ax.bar(x, values_of, width, label='OF', color='b')
    ax.set_ylabel('Average')
    labels = ['TP', 'FP', 'TN', 'FN']
    ax.set_title('Model Evaluation Average Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # # Add the values to the bars
    # for i, v in enumerate(values_of):
    #     ax.text(i, v + 0.01, str(round(v, 2)), color='black', ha='center')

    values_cf = [avg_TP_cf, avg_FP_cf, avg_TN_cf, avg_FN_cf]
    ax.bar(x + width, values_cf, width, label='CF', color='r')
    ax.set_xticks(x)
    ax.legend()
    plt.show()


    # Now do a box plot for the values
    fig, ax = plt.subplots()
    x = np.arange(9)
    values = [
        TP_values_of, 
        TP_values_cf,
        FP_values_of,
        FP_values_cf,
        TN_values_of,
        TN_values_cf,
        FN_values_of,
        FN_values_cf]
    
    labels = [' ', 'TP OF', 'TP CF', 'FP OF', 'FP CF', 'TN OF', 'TN CF', 'FN OF', 'FN CF']
    ax.boxplot(values, showmeans=True, notch=True)
    ax.set_title('Confusion Matrix Model Evaluation Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.tight_layout()
    plt.show()

def plot_ba_mcc(eval_results: dict) -> None:
    """Plot the BA and MCC scores of the evaluation results."""
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

def plot_itrs(eval_results: dict) -> None:
    """Plot the ITRs of the evaluation results."""
    
    itr_cf = []
    itr_of = []

    for user_id, user_data in eval_results.items():
        itr_of.append(user_data['of']['ITR'])
        itr_cf.append(user_data['cf']['ITR'])

    # plot the results with a bar chart (0-1) and add labels with the values
    fig, ax = plt.subplots()
    x = np.arange(2)
    width = 0.35
    itr_values = [np.mean(itr_of), np.mean(itr_cf)]
    labels = ['ITR OF', 'ITR CF']
    ax.bar(x, itr_values, width, label='Scores')
    ax.set_ylabel('Scores')
    ax.set_title('Model Evaluation ITR Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # Add the values to the bars
    for i, v in enumerate(itr_values):
        ax.text(i, v + 0.01, str(round(v, 2)), color='black', ha='center')
    
    plt.show()

    # box plot the results
    fig, ax = plt.subplots()
    x = np.arange(3)
    values = [itr_of, itr_cf]
    labels = [' ', 'ITR OF', 'ITR CF']
    ax.boxplot(values, showmeans=True, notch=True)
    ax.set_title('ITR Model Evaluation Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.show()

    return itr_of, itr_cf

def plot_boxplot_evaluation_results(eval_results, run_stats=False) -> None:
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
    x = np.arange(3)
    ba_values = [ba_values_of, ba_values_cf]
    labels = [' ', 'BA OF', 'BA CF']
    ax.boxplot(ba_values, showmeans=True, notch=True)
    ax.set_title('BA Model Evaluation Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.show()

    # plot a box plot for the MCC
    fig, ax = plt.subplots()
    x = np.arange(3)
    mcc_values = [mcc_values_of, mcc_values_cf]
    labels = [' ', 'MCC OF', 'MCC CF']
    ax.boxplot(mcc_values, showmeans=True, notch=True)
    ax.set_title('MCC Model Evaluation Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.show()

    return ba_values_of, ba_values_cf, mcc_values_of, mcc_values_cf


def run_ttest(ba_values_of, ba_values_cf, mcc_values_of, mcc_values_cf, itr_of, itr_cf) -> tuple:
    """Run a t-test on the values"""
    # # run a t-test to see if the scores are significantly different
    from scipy.stats import ttest_rel
    ba_t, ba_p = ttest_rel(ba_values_of, ba_values_cf)
    print(f"BA T-Test: {ba_t}, {ba_p}")
    mcc_t, mcc_p = ttest_rel(mcc_values_of, mcc_values_cf)
    print(f"MCC T-Test: {mcc_t}, {mcc_p}")
    itr_t, itr_p = ttest_rel(itr_of, itr_cf)
    print(f"ITR T-Test: {itr_t}, {itr_p}")
    return (ba_t, ba_p), (mcc_t, mcc_p), (itr_t, itr_p)


if __name__ in "__main__":
    path = ask_filename('*json')
    data = load_json_file(path)
    results = extract_and_score_model_output(data)
    save_results(results, 'score_results_2.json')
    eval_results = evaluate_model_output(results)
    save_results(eval_results, 'eval_results_2.json')
    plot_average_evaluation_results(eval_results)
    ba_mcc_results = plot_boxplot_evaluation_results(eval_results)
    itr_results = plot_itrs(eval_results)
    run_ttest(*ba_mcc_results, *itr_results)
    # breakpoint()