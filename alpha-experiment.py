from pathlib import Path

import numpy as np
import pywt

from bcipy.helpers.load import load_raw_data
from bcipy.helpers.triggers import trigger_decoder
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import get_default_transform

from bcipy.helpers.load import load_json_parameters

from bcipy.helpers.visualization import generate_offline_analysis_screen

import matplotlib.pyplot as plt


def cwt(data, freq: int, fs: int):
    # Original data.shape == (channels, trials, samples)
    # Want: (trials, channels, samples)
    data = data.transpose(1, 0, 2)
    wavelet = "cmor5.0-1.0"  # "morl"  # TODO - important choice of hyperparams here
    scales = pywt.central_frequency(wavelet) * fs / np.array(freq)
    all_coeffs = []
    for trial in data:
        coeffs, _ = pywt.cwt(trial, scales, wavelet)  # shape == (scales, channels, time)
        all_coeffs.append(coeffs)

    final_data = np.stack(all_coeffs)
    if np.any(np.iscomplex(final_data)):
        final_data = np.abs(final_data) ** 2

    # have shape == (trials, freqs, channels, time)
    # want shape == (trials, freqs*channels, time)
    final_data = final_data.reshape(final_data.shape[0], -1, final_data.shape[-1])
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
    return model_performance


def make_plots(data, labels):
    """
    targets = data[labels]
    """
    breakpoint()


def main(data_path, parameters):
    data, labels, fs = load_data(data_path, parameters)
    data = cwt(data, args.selected_freq, fs)
    print(data.shape)
    # data.shape == (trials, channels, samples)
    # each trial has [-1.25s, 1.25s]

    # data begins at -1250, want to slice all trials from -600 to -100
    begin = int(0.65 * fs)  # .650 s * N samples/s = samples   # TODO - don't hardcode
    duration = int(0.5 * fs)
    end = begin + duration
    print(f"Baseline region (in samples, not seconds): {begin=}, {duration=}, {end=}")
    means = data[..., begin:end].mean(axis=2, keepdims=True)  # (trials * 4) means
    stdevs = data[..., begin:end].std(axis=2, keepdims=True)  # (trials * 4) stdevs

    # z-score the data in the window of interest [300, 800]
    # TODO - this assumes each trial starts at -1250ms, not yet true!
    start_s = 1.25 + 0.3
    begin = int(start_s * fs)
    duration = int(0.5 * fs)
    end = begin + duration
    print(f"Response region (in samples, not seconds): {begin=}, {duration=}, {end=}")
    print(data.min(), data.mean(), data.max())
    data = data[..., begin:end] - means
    data = data / stdevs
    print(data.min(), data.mean(), data.max())

    # make_plots(data, labels)

    # NOTE - model expects (channels, trials, samples)
    data = data.transpose([1, 0, 2])

    generate_offline_analysis_screen(
        data,
        labels,
        model=None,
        folder=data_path,
        save_figure=True,
        # down_sample_rate=2,
        fs=fs,
        plot_x_ticks=8,
        plot_average=False,
        show_figure=False,
        channel_names=None,
    )

    model_performance = fit_model(data, labels)
    print(model_performance.auc)


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
    - Set CWT parameters so that data matches BrainVision - try plotting the full (-1250, 1250) trials, averaged for target and nontarget
    - Can try export data from BrainVision and train model
    - Try with/without PCA step
    """

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, help="Path to data folder", required=True)
    p.add_argument("--selected-freq", type=int, help="Frequency to keep after CWT", default=10)
    p.add_argument("-p", "--parameters_file", default="bcipy/parameters/parameters.json")
    args = p.parse_args()

    if not args.data.exists():
        raise ValueError("data path does not exist")

    print(f"Input data folder: {str(args.data)}")
    print(f"Selected freq: {str(args.selected_freq)}")
    print(f"Loading params from {args.parameters_file}")
    parameters = load_json_parameters(args.parameters_file, value_cast=True)
    main(args.data, parameters)
