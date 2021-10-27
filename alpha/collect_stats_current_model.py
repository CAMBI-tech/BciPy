import logging
from pathlib import Path

import numpy as np
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import load_json_parameters, load_raw_data
from bcipy.helpers.triggers import trigger_decoder
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import get_default_transform
from sklearn.metrics import balanced_accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")


def reorder(data):
    return data.transpose(1, 0, 2)


def main(input_path, parameters):
    # extract relevant session information from parameters file
    trial_length = 0.5
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
    k_folds = parameters.get("k_folds", 10)

    # Load raw data
    raw_data = load_raw_data(Path(input_path, raw_data_file))
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
    _, t_t_i, t_i, offset = trigger_decoder(mode="calibration", trigger_path=f"{input_path}/{triggers_file}")

    offset = offset + static_offset

    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    # channel_names = ["P4", "Fz", "Pz", "F7", "PO8", "PO7", "Oz"]
    # channel_map = [0, 0, 1, 0, 1, 1, 1, 0]
    channel_map = analysis_channels(channels, type_amp)
    model = make_pipeline(FunctionTransformer(reorder), PcaRdaKdeModel(k_folds=k_folds))
    data, labels = PcaRdaKdeModel.reshaper(
        trial_labels=t_t_i,
        timing_info=t_i,
        eeg_data=data,
        fs=fs,
        trials_per_inquiry=parameters.get("stim_length"),
        offset=offset,
        channel_map=channel_map,
        trial_length=trial_length,
    )
    np.random.seed(1)

    data = data.transpose(1, 0, 2)

    # # Manual version, in case of sanity checking:
    # x_train, x_test, y_train, y_test = train_test_split(data, labels)
    # model.fit(x_train, y_train)
    # print(model.steps[1][1].evaluate(x_test.transpose(1, 0, 2), y_test))

    results = cross_validate(
        model,
        data,
        labels,
        cv=10,
        n_jobs=-1,
        return_train_score=True,
        scoring={"balanced_accuracy": make_scorer(balanced_accuracy_score), "roc_auc": make_scorer(roc_auc_score)},
    )

    avg_train_acc = round(results["train_balanced_accuracy"].mean(), 3)
    std_train_acc = round(results["train_balanced_accuracy"].std(), 3)

    avg_test_acc = round(results["test_balanced_accuracy"].mean(), 3)
    std_test_acc = round(results["test_balanced_accuracy"].std(), 3)

    avg_train_auc = round(results["train_roc_auc"].mean(), 3)
    std_train_auc = round(results["train_roc_auc"].std(), 3)

    avg_test_auc = round(results["test_roc_auc"].mean(), 3)
    std_test_auc = round(results["test_roc_auc"].std(), 3)

    avg_fit_time = round(results["fit_time"].mean(), 3)
    std_fit_time = round(results["fit_time"].std(), 3)

    avg_score_time = round(results["score_time"].mean(), 3)
    std_score_time = round(results["score_time"].std(), 3)

    print(
        {
            "avg_fit_time": avg_fit_time,
            "std_fit_time": std_fit_time,
            "avg_score_time": avg_score_time,
            "std_score_time": std_score_time,
            "avg_train_roc_auc": avg_train_auc,
            "std_train_roc_auc": std_train_auc,
            "avg_test_roc_auc": avg_test_auc,
            "std_test_roc_auc": std_test_auc,
            "avg_train_balanced_accuracy": avg_train_acc,
            "std_train_balanced_accuracy": std_train_acc,
            "avg_test_balanced_accuracy": avg_test_acc,
            "std_test_balanced_accuracy": std_test_acc,
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", default=None)
    parser.add_argument("-p", "--parameters_file", default="bcipy/parameters/parameters.json")
    args = parser.parse_args()

    log.info(f"Loading params from {args.parameters_file}")
    parameters = load_json_parameters(args.parameters_file, value_cast=True)
    main(args.data_folder, parameters)
    log.info("Offline Analysis complete.")
