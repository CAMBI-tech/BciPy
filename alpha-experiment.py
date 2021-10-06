import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
from loguru import logger
from pyriemann.classification import TSclassifier
from pyriemann.estimation import Covariances
from rich.console import Console
from rich.table import Table
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from bcipy.helpers.load import load_json_parameters, load_raw_data
from bcipy.helpers.triggers import trigger_decoder
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import get_default_transform


def cwt(data, freq: int, fs: int):
    # Original data.shape == (channels, trials, samples)
    # Want: (trials, channels, samples)
    data = data.transpose(1, 0, 2)
    wavelet = "cmor1.5-1.0"  # "morl"  # TODO - important choice of hyperparams here
    # wavelet = "morl"  # "morl"  # TODO - important choice of hyperparams here
    scales = pywt.central_frequency(wavelet) * fs / np.array(freq)
    all_coeffs = []
    for trial in data:
        coeffs, _ = pywt.cwt(trial, scales, wavelet)  # shape == (scales, channels, time)
        all_coeffs.append(coeffs)

    final_data = np.stack(all_coeffs)
    if np.any(np.iscomplex(final_data)):
        print("Converting complex to real")
        final_data = np.abs(final_data) ** 2

    # have shape == (trials, freqs, channels, time)
    # want shape == (trials, freqs*channels, time)
    final_data = final_data.reshape(final_data.shape[0], -1, final_data.shape[-1])

    # now put channels in front
    final_data = final_data.transpose([1, 0, 2])
    return final_data


def load_data(data_folder: Path, parameters: dict):
    # from offline analysis

    # extract relevant session information from parameters file
    trial_length = 2.5
    triggers_file = parameters.get("trigger_file_name", "triggers.txt")
    raw_data_file = parameters.get("raw_data_name", "raw_data.csv")

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate", 2)
    notch_filter = parameters.get("notch_filter_frequency", 60)
    hp_filter = parameters.get("filter_high", 45)
    lp_filter = parameters.get("filter_low", 2)
    filter_order = parameters.get("filter_order", 2)

    # get offset and k folds
    static_offset = parameters.get("static_trigger_offset", 0)
    pre_stim_offset = -1.25  # NOTE - want each trial to start 1.25s BEFORE stim

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    fs = raw_data.sample_rate

    print(f"Channels read from csv: {channels}")
    print(f"Device type: {type_amp}")

    default_transform = get_default_transform(
        sample_rate_hz=fs,
        notch_freq_hz=notch_filter,
        bandpass_low=lp_filter,
        bandpass_high=hp_filter,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    data, fs = default_transform(raw_data.by_channel(), fs)

    # Process triggers.txt
    _, t_t_i, t_i, offset = trigger_decoder(mode="calibration", trigger_path=f"{data_folder}/{triggers_file}")

    offset = offset + static_offset + pre_stim_offset

    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    # channel_names = ["P4", "Fz", "Pz", "F7", "PO8", "PO7", "Oz"]
    channel_map = [0, 0, 1, 0, 1, 1, 1, 0]

    # TODO - need to extract trials including some amt of time BEFORE stim

    data, labels = PcaRdaKdeModel.reshaper(
        trial_labels=t_t_i,
        timing_info=t_i,
        eeg_data=data,
        fs=fs,
        trials_per_inquiry=10,
        offset=offset,
        channel_map=channel_map,
        trial_length=trial_length,
    )

    return data, labels, fs


def fit_model(data, labels):
    k_folds = 10
    model = PcaRdaKdeModel(k_folds=k_folds)
    print("Training model. This will take some time...")
    model.fit(data, labels)
    model_performance = model.evaluate(data, labels)
    print(model_performance.auc)
    return model_performance


def _fit(data, labels, n_folds, flatten_data, clf):
    np.random.seed(1)

    data = data.transpose([1, 0, 2])
    if flatten_data:
        data = data.reshape([data.shape[0], -1])

    results = cross_validate(
        clf,
        data,
        labels,
        cv=n_folds,
        n_jobs=-1,
        return_train_score=True,
        scoring=["balanced_accuracy", "roc_auc"],
    )

    return {
        "avg_fit_time": round(results["fit_time"].mean(), 3),
        "avg_score_time": round(results["score_time"].mean(), 3),
        "avg_train_roc_auc": round(results["train_roc_auc"].mean(), 3),
        "avg_test_roc_auc": round(results["test_roc_auc"].mean(), 3),
        "avg_train_balanced_accuracy": round(results["train_balanced_accuracy"].mean(), 3),
        "avg_test_balanced_accuracy": round(results["test_balanced_accuracy"].mean(), 3),
    }


def fit_mlp(data, labels, test_frac=0.1):
    return _fit(data, labels, test_frac, model_class=MLPClassifier)


def fit_svm(data, labels, test_frac=0.1):
    return _fit(data, labels, test_frac, model_class=partial(SVC, probability=True))
    # return _fit(data, labels, test_frac, model_class=partial(SVC, probability=True, gamma=2, C=1))


def fit_qda(data, labels, test_frac=0.1):
    return _fit(data, labels, test_frac, model_class=QuadraticDiscriminantAnalysis)


def fit_logr(data, labels, test_frac=0.1):
    return _fit(data, labels, test_frac, model_class=LogisticRegression)


def make_plots(data, labels, filename):
    targets_data = data[:, labels == 1, :]
    nontargets_data = data[:, labels == 0, :]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(targets_data.mean(axis=(0, 1)), c="r")  # average across channels and trials
    ax.plot(nontargets_data.mean(axis=(0, 1)), c="b")
    fig.savefig(filename, bbox_inches="tight", dpi=300)


def main(input_path, output_path, parameters):
    # Load data and CWT preprocess
    data, labels, fs = load_data(input_path, parameters)
    make_plots(data, labels, output_path / "0.raw_data.png")
    data = cwt(data, args.freq, fs)
    make_plots(data, labels, output_path / "1.cwt_data.png")
    print(data.shape)
    # data.shape == (trials, channels, samples)
    # each trial has [-1.25s, 1.25s]

    # Normalize data
    # data begins at -1250, want to slice all trials from -600 to -100
    begin = int(0.65 * fs)  # .650 s * N samples/s = samples   # TODO - don't hardcode
    duration = int(0.5 * fs)
    end = begin + duration
    print(f"Baseline region (in samples, not seconds): {begin=}, {duration=}, {end=}")
    means = data[..., begin:end].mean(axis=(1, 2), keepdims=True)  # (4) means
    stdevs = data[..., begin:end].std(axis=(1, 2), keepdims=True)  # (4) stdevs

    # z-score the data in the window of interest [300, 800]
    # TODO - this assumes each trial starts at -1250ms, not yet true!
    start_s = 1.25 + 0.3
    begin = int(start_s * fs)
    duration = int(0.5 * fs)
    end = begin + duration
    print(f"Response region (in samples, not seconds): {begin=}, {duration=}, {end=}")
    print(data.min(), data.mean(), data.max())
    # The copy we care about for modeling
    z_transformed_target_window = (data[..., begin:end] - means) / stdevs
    # Copy of entire window for plotting
    z_transformed_entire_data = (data - means) / stdevs
    print(z_transformed_target_window.min(), z_transformed_target_window.mean(), z_transformed_target_window.max())

    # NOTE - model expects (channels, trials, samples)
    make_plots(z_transformed_target_window, labels, output_path / "2.z_target_window.png")
    make_plots(z_transformed_entire_data, labels, output_path / "3.z_entire_data.png")

    # RDA/KDE (no PCA step)
    # print(fit_model(z_transformed_target_window, labels).auc)

    ts_logr = make_pipeline(Covariances(), TSclassifier(clf=LogisticRegression(class_weight="balanced")))

    reports = []
    lr_kw = {"max_iter": 200, "solver": "liblinear"}
    for model_name, uses_balanced_training, flatten_data, clf in [
        ("Multi-layer Perceptron", False, True, MLPClassifier(max_iter=500)),
        ("Support Vector Classifier", True, True, SVC(probability=True, class_weight="balanced")),
        ("Support Vector Classifier", False, True, SVC(probability=True)),
        ("QuadraticDiscriminantAnalysis", True, True, QuadraticDiscriminantAnalysis()),
        ("LogisticRegression", True, True, LogisticRegression(class_weight="balanced", **lr_kw)),
        ("LogisticRegression", False, True, LogisticRegression(**lr_kw)),
        ("Decision Tree", True, True, DecisionTreeClassifier(class_weight="balanced")),
        ("Decision Tree", False, True, DecisionTreeClassifier()),
        ("Random Forest", True, True, RandomForestClassifier(class_weight="balanced")),
        ("Random Forest", False, True, RandomForestClassifier()),
        ("K-nearest neighbors", False, True, KNeighborsClassifier(10)),
        ("K-nearest neighbors, distance-weighted", False, True, KNeighborsClassifier(10, weights="distance")),
        ("Tangent Space, Logistic Regression", True, False, ts_logr),
    ]:
        n_folds = 10
        print("Run model class:", model_name)
        report = _fit(data, labels, n_folds, flatten_data, clf)
        report["name"] = model_name
        report["uses_balanced_training"] = uses_balanced_training
        reports.append(report)

    table = Table(title=f"Alpha Classifier Comparison ({n_folds}-fold cross validation)")
    table.add_column("Model Name", style="red")
    table.add_column("Train with balanced classes?", style="orange1")
    table.add_column("Avg fit time", style="yellow"),
    table.add_column("Avg score time", style="green"),
    table.add_column("Avg train roc auc", style="blue"),
    table.add_column("Avg test roc auc", style="magenta"),
    table.add_column("Avg train balanced accuracy", style="red"),
    table.add_column("Avg test balanced accuracy", style="orange1"),

    for report in reports:
        report = {k: str(v) for k, v in report.items()}
        table.add_row(
            report["name"],
            report["uses_balanced_training"],
            report["avg_fit_time"],
            report["avg_score_time"],
            report["avg_train_roc_auc"],
            report["avg_test_roc_auc"],
            report["avg_train_balanced_accuracy"],
            report["avg_test_balanced_accuracy"],
        )

    console = Console(record=True)
    console.print(table)
    console.save_html(output_path / "results.html")


if __name__ == "__main__":
    """
    Overall workflow:
        raw data:
            7 channel, exclude F7 and Fcz, include Pz, Oz, Po7, Po8
            -1250ms to 1250ms around stim

        pick the 4 channels of interest

        CWT:
            give a parameter for the single wavelet frequency/scale to keep at the end (default ~10)
            consider overlap with SSVEP response (or its harmonics)

        -600ms to -100ms window for "baseline"
            get mean, stdev

        for each point in [300, 800]:
            z-score each sample point using the mean & stdev above

        pca/rda/kde

    TODO:
    - Hyperparams:
        - when converting from complex, square or not?
        - for z-scoring, per channel, or per trial, etc
        - wavelet and its params
        - Set CWT parameters so that data matches BrainVision - try plotting the full (-1250, 1250) trials, averaged for target and nontarget

    - switch to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    - Include the PcaRdaKdeModel and RdaKdeModel
    - fine-tune the choice of models
    - Setting up a quick run script to run the code on each participant using the right wavelet frequency

    - Can try export data from BrainVision and train model
    """

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, help="Path to data folder", required=True)
    p.add_argument("--output", type=Path, help="Path to save outputs", required=True)
    p.add_argument("--freq", type=float, help="Frequency to keep after CWT", default=10)
    p.add_argument("--parameters_file", default="bcipy/parameters/parameters.json")
    args = p.parse_args()

    if not args.input.exists():
        raise ValueError("data path does not exist")

    args.output.mkdir(exist_ok=True, parents=True)

    print(f"Input data folder: {str(args.input)}")
    print(f"Selected freq: {str(args.freq)}")
    print(f"Loading params from {args.parameters_file}")
    parameters = load_json_parameters(args.parameters_file, value_cast=True)
    with logger.catch(onerror=lambda _: sys.exit(1)):
        main(args.input, args.output, parameters)
