# mypy: disable-error-code="assignment,var-annotated"
import numpy as np
from sklearn.utils import resample
from typing import List, Tuple
import logging
from tqdm import tqdm

from bcipy.config import (TRIGGER_FILENAME, SESSION_LOG_FILENAME)
from bcipy.helpers.acquisition import analysis_channels
from bcipy.core.raw_data import RawData
from bcipy.core.stimuli import update_inquiry_timing
from bcipy.core.triggers import TriggerType, trigger_decoder
from bcipy.preferences import preferences
from bcipy.signal.model.base_model import SignalModelMetadata
from bcipy.signal.model.base_model import SignalModel
from bcipy.signal.model.gaussian_mixture import GaussianProcess
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import (ERPTransformParams, extract_eye_info,
                                  filter_inquiries, get_default_transform)
from bcipy.acquisition.devices import DeviceSpec
from bcipy.core.parameters import Parameters


logger = logging.getLogger(SESSION_LOG_FILENAME)


def calculate_eeg_gaze_fusion_acc(
        eeg_data: RawData,
        gaze_data: RawData,
        device_spec_eeg: DeviceSpec,
        device_spec_gaze: DeviceSpec,
        symbol_set: List[str],
        parameters: Parameters,
        data_folder: str,
        n_iterations: int = 10,
        eeg_model: SignalModel = PcaRdaKdeModel,
        gaze_model: SignalModel = GaussianProcess) -> Tuple[List[float], List[float], List[float]]:
    """
    Preprocess the EEG and gaze data. Calculate the accuracy of the fusion of EEG and Gaze models.
    Args:
        eeg_data: Raw EEG data. Test data will be extracted and selected along with gaze data.
        gaze_data: Raw Gaze data.
        device_spec_eeg: Device specification for EEG data.
        device_spec_gaze: Device specification for Gaze data.
        symbol_set: Set of symbols used in the experiment. (Default = alphabet())
        parameters: Parameters file containing the experiment-specific parameters.
        data_folder: Folder containing the raw data and the results.
        n_iterations: Number of iterations to bootstrap the accuracy calculation. (Default = 10)
        eeg_model: EEG model to use for the fusion. (Default = PcaRdaKdeModel)
        gaze_model: Gaze model to use for the fusion. (Default = GaussianProcess)
    Returns:
        eeg_acc: accuracy of the EEG model only
        gaze_acc: accuracy of the gaze model only
        fusion_acc: accuracy of the fusion
    """
    logger.info(f"Calculating EEG [{eeg_model.name}] and Gaze [{gaze_model.name}] model fusion accuracy.")
    # Extract relevant session information from parameters file
    trial_window = parameters.get("trial_window", (0.0, 0.5))
    window_length = trial_window[1] - trial_window[0]  # eeg window length, in seconds

    prestim_length = parameters.get("prestim_length")
    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)

    # Get signal filtering information
    transform_params: ERPTransformParams = parameters.instantiate(ERPTransformParams)
    downsample_rate = transform_params.down_sampling_rate
    static_offset = device_spec_eeg.static_offset

    # Get the flash time (for gaze analysis)
    flash_time = parameters.get("time_flash")

    eeg_channels = eeg_data.channels
    eeg_channel_map = analysis_channels(eeg_channels, device_spec_eeg)
    eeg_sample_rate = eeg_data.sample_rate
    gaze_sample_rate = gaze_data.sample_rate

    # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=eeg_sample_rate,
        notch_freq_hz=transform_params.notch_filter_frequency,
        bandpass_low=transform_params.filter_low,
        bandpass_high=transform_params.filter_high,
        bandpass_order=transform_params.filter_order,
        downsample_factor=transform_params.down_sampling_rate,
    )

    # Define the model object before reshaping the data
    k_folds = parameters.get("k_folds")
    eeg_model = eeg_model(k_folds=k_folds)
    # Select between the two (or three) gaze models to test:
    gaze_model = gaze_model()

    # Process triggers.txt files for eeg data:
    trigger_targetness, trigger_timing, inquiry_symbols = trigger_decoder(
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        remove_pre_fixation=True,
        offset=static_offset,
        exclusion=[
            TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )

    # Same as above, but with the 'prompt' triggers added for gaze analysis:
    trigger_targetness_gaze, trigger_timing_gaze, trigger_symbols = trigger_decoder(
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        remove_pre_fixation=False,
        exclusion=[
            TriggerType.PREVIEW,
            TriggerType.EVENT,
            TriggerType.FIXATION,
            TriggerType.SYSTEM,
            TriggerType.OFFSET],
        device_type='EYETRACKER',
        apply_starting_offset=False
    )
    ''' Trigger_timing includes PROMPT and excludes FIXATION '''

    target_symbols = [trigger_symbols[idx]
                      for idx, targetness in enumerate(trigger_targetness_gaze) if targetness == 'prompt']
    total_len = trials_per_inquiry + 1    # inquiry length + the prompt symbol
    inq_start = trigger_timing_gaze[1::total_len]  # inquiry start times, exluding prompt and fixation

    # update the trigger timing list to account for the initial trial window
    corrected_trigger_timing = [timing + trial_window[0] for timing in trigger_timing]

    erp_data, _fs_eeg = eeg_data.by_channel()
    trajectory_data, _fs_eye = gaze_data.by_channel()

    # Reshaping EEG data:
    eeg_inquiries, eeg_inquiry_labels, eeg_inquiry_timing = eeg_model.reshaper(
        trial_targetness_label=trigger_targetness,
        timing_info=corrected_trigger_timing,
        eeg_data=erp_data,
        sample_rate=eeg_sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=eeg_channel_map,
        poststimulus_length=window_length,
        prestimulus_length=prestim_length,
        transformation_buffer=buffer,
    )
    # Size = Inquiries x Channels x Samples

    # Reshaping gaze data:
    gaze_inquiries_dict, gaze_inquiries_list, _ = gaze_model.reshaper(
        inq_start_times=inq_start,
        target_symbols=target_symbols,
        gaze_data=trajectory_data,
        sample_rate=gaze_sample_rate,
        stimulus_duration=flash_time,
        num_stimuli_per_inquiry=trials_per_inquiry,
        symbol_set=symbol_set
    )

    # More EEG preprocessing:
    eeg_inquiries, fs = filter_inquiries(eeg_inquiries, default_transform, eeg_sample_rate)
    eeg_inquiry_timing = update_inquiry_timing(eeg_inquiry_timing, downsample_rate)
    trial_duration_samples = int(window_length * fs)

    # More gaze preprocessing:
    inquiry_length = gaze_inquiries_list[0].shape[1]  # number of time samples in each inquiry
    predefined_dimensions = 4  # left_x, left_y, right_x, right_y
    preprocessed_gaze_data = np.zeros((len(gaze_inquiries_list), predefined_dimensions, inquiry_length))
    # Extract left_x, left_y, right_x, right_y for each inquiry
    for j in range(len(gaze_inquiries_list)):
        left_eye, right_eye, _, _, _, _ = extract_eye_info(gaze_inquiries_list[j])
        preprocessed_gaze_data[j] = np.concatenate((left_eye.T, right_eye.T,), axis=0)

    preprocessed_gaze_dict = {i: [] for i in symbol_set}
    for i in symbol_set:
        # Skip if there's no evidence for this symbol:
        if len(gaze_inquiries_dict[i]) == 0:
            continue
        for j in range(len(gaze_inquiries_dict[i])):
            left_eye, right_eye, _, _, _, _ = extract_eye_info(gaze_inquiries_dict[i][j])
            preprocessed_gaze_dict[i].append((np.concatenate((left_eye.T, right_eye.T), axis=0)))
        preprocessed_gaze_dict[i] = np.array(preprocessed_gaze_dict[i])

    # Find the time averages for each symbol:
    centralized_data_dict = {i: [] for i in symbol_set}
    time_average_per_symbol = {i: [] for i in symbol_set}
    for sym in symbol_set:
        # Skip if there's no evidence for this symbol:
        try:
            if len(gaze_inquiries_dict[sym]) == 0:
                continue
        except BaseException:
            continue

        for j in range(len(preprocessed_gaze_dict[sym])):
            temp = np.mean(preprocessed_gaze_dict[sym][j], axis=1)
            time_average_per_symbol[sym].append(temp)
            centralized_data_dict[sym].append(
                gaze_model.subtract_mean(
                    preprocessed_gaze_dict[sym][j],
                    temp))  # Delta_t = X_t - mu
        centralized_data_dict[sym] = np.array(centralized_data_dict[sym])
        time_average_per_symbol[sym] = np.mean(np.array(time_average_per_symbol[sym]), axis=0)

    # Take the time average of the gaze data:
    centralized_gaze_data = np.zeros_like(preprocessed_gaze_data)
    for i, (_, sym) in enumerate(zip(preprocessed_gaze_data, target_symbols)):
        centralized_gaze_data[i] = gaze_model.subtract_mean(preprocessed_gaze_data[i], time_average_per_symbol[sym])

    """
    Calculate the accuracy of the fusion of EEG and Gaze models. Use the number of iterations to change bootstraping.
    """
    eeg_acc = []
    gaze_acc = []
    fusion_acc = []
    # selection length is the length of eeg or gaze data, whichever is smaller:
    selection_length = min(len(eeg_inquiries[1]), len(preprocessed_gaze_data))

    progress_bar = tqdm(
        range(n_iterations),
        total=n_iterations,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [est. {remaining}][ela. {elapsed}]\n",
        colour='MAGENTA')
    for _progress in progress_bar:
        progress_bar.set_description(f"Running iteration {_progress + 1}/{n_iterations}")
        # Pick a train and test dataset (that consists of non-train elements) until test dataset is not empty:
        train_indices = resample(list(range(selection_length)), replace=True, n_samples=100)
        test_indices = np.array([x for x in list(range(selection_length)) if x not in train_indices])
        if len(test_indices) == 0:
            break

        train_data_eeg = eeg_inquiries[:, train_indices, :]
        test_data_eeg = eeg_inquiries[:, test_indices, :]
        eeg_inquiry_timing = np.array(eeg_inquiry_timing)
        train_eeg_inquiry_timing = eeg_inquiry_timing[train_indices]
        test_eeg_inquiry_timing = eeg_inquiry_timing[test_indices]
        inquiry_symbols_test = np.array([])
        for t_i in test_indices:
            inquiry_symbols_test = np.append(inquiry_symbols_test,
                                             inquiry_symbols[t_i * trials_per_inquiry:(t_i + 1) * trials_per_inquiry])
        inquiry_symbols_test = inquiry_symbols_test.tolist()

        # Now extract the inquiries from trials for eeg model fitting:
        preprocessed_train_eeg = eeg_model.reshaper.extract_trials(
            train_data_eeg, trial_duration_samples, train_eeg_inquiry_timing)
        preprocessed_test_eeg = eeg_model.reshaper.extract_trials(
            test_data_eeg, trial_duration_samples, test_eeg_inquiry_timing)

        # train and save the eeg model a pkl file
        # Flatten the labels (0=nontarget/1=target) prior to model fitting
        erp_train_labels = eeg_inquiry_labels[train_indices].flatten().tolist()
        # erp_test_labels = eeg_inquiry_labels[test_indices].flatten().tolist()
        eeg_model.fit(preprocessed_train_eeg, erp_train_labels)
        eeg_model.metadata = SignalModelMetadata(device_spec=device_spec_eeg,
                                                 transform=default_transform)
        # save_model(eeg_model, Path(data_folder, f"model_{eeg_model.auc:0.4f}.pkl"))
        preferences.signal_model_directory = data_folder

        # extract train and test indices for gaze data:
        centralized_gaze_data_train = centralized_gaze_data[train_indices]
        # gaze_train_labels = np.array([target_symbols[i] for i in train_indices])
        gaze_data_test = preprocessed_gaze_data[test_indices]         # test set is NOT centralized
        gaze_test_labels = np.array([target_symbols[i] for i in test_indices])
        # generate a tuple that matches the index of the symbol with the symbol itself:
        symbol_to_index = {symbol: i for i, symbol in enumerate(symbol_set)}

        reshaped_data = centralized_gaze_data_train.reshape(
            (len(centralized_gaze_data_train), inquiry_length * predefined_dimensions))
        units = 1e4
        reshaped_data *= units
        cov_matrix = np.cov(reshaped_data, rowvar=False)
        time_horizon = 9

        for eye_coord_0 in range(predefined_dimensions):
            for eye_coord_1 in range(predefined_dimensions):
                for time_0 in range(inquiry_length):
                    for time_1 in range(inquiry_length):
                        l_ind = eye_coord_0 * inquiry_length + time_0
                        m_ind = eye_coord_1 * inquiry_length + time_1
                        if np.abs(time_1 - time_0) > time_horizon:
                            cov_matrix[l_ind, m_ind] = 0

        reshaped_mean = np.mean(reshaped_data, axis=0)
        eps = 0
        regularized_cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * eps
        try:
            inv_cov_matrix = np.linalg.inv(regularized_cov_matrix)
        except BaseException:
            # Singular matrix, using pseudo-inverse instead
            eps = 10e-3  # add a small value to the diagonal to make the matrix invertible
            inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(len(cov_matrix)) * eps)
            # inv_cov_matrix = np.linalg.pinv(cov_matrix + np.eye(len(cov_matrix))*eps)
        denominator_gaze = 0

        # Given the test data, compute the log likelihood ratios for each symbol,
        # from eeg and gaze models:
        eeg_log_likelihoods = np.zeros((len(gaze_data_test), (len(symbol_set))))
        gaze_log_likelihoods = np.zeros((len(gaze_data_test), (len(symbol_set))))

        # Save the max posterior and the second max posterior for each test point:
        target_posteriors_gaze = np.zeros((len(gaze_data_test), 2))
        target_posteriors_eeg = np.zeros((len(gaze_data_test), 2))
        target_posteriors_fusion = np.zeros((len(gaze_data_test), 2))

        counter_gaze = 0
        counter_eeg = 0
        counter_fusion = 0
        counter_test_samples = len(gaze_data_test)
        for test_idx, test_data in enumerate(gaze_data_test):
            numerator_gaze_list = []
            diff_list = []
            # skip the analysis for this data point if there is no training example from the symbol:
            if time_average_per_symbol[gaze_test_labels[test_idx]] == []:
                counter_test_samples -= 1
                print(f"Skipping the test case with symbol {gaze_test_labels[test_idx]}, no training data available.")
                continue

            for idx, sym in enumerate(symbol_set):
                # skip if there is no training example from the symbol
                if time_average_per_symbol[sym] == []:
                    gaze_log_likelihoods[test_idx, idx] = -100000  # set a very small value
                else:
                    central_data = gaze_model.subtract_mean(test_data, time_average_per_symbol[sym])
                    flattened_data = central_data.reshape((inquiry_length * predefined_dimensions,))
                    flattened_data *= units
                    diff = flattened_data - reshaped_mean
                    diff_list.append(diff)
                    numerator = -np.dot(diff.T, np.dot(inv_cov_matrix, diff)) / 2
                    numerator_gaze_list.append(numerator)
                    unnormalized_log_likelihood_gaze = numerator - denominator_gaze
                    gaze_log_likelihoods[test_idx, idx] = unnormalized_log_likelihood_gaze
            normalized_posterior_gaze_only = gaze_log_likelihoods[test_idx, :] - np.log(np.sum(np.exp(gaze_log_likelihoods[test_idx, :])))
            # Find the max likelihood:
            max_like_gaze = np.argmax(normalized_posterior_gaze_only)

            posterior_of_true_label_gaze = normalized_posterior_gaze_only[symbol_to_index[gaze_test_labels[test_idx]]]
            top_competitor_gaze = np.sort(normalized_posterior_gaze_only)[-2]
            target_posteriors_gaze[test_idx, 0] = posterior_of_true_label_gaze
            target_posteriors_gaze[test_idx, 1] = top_competitor_gaze
            # Check if it's the same as the target
            if symbol_set[max_like_gaze] == gaze_test_labels[test_idx]:
                counter_gaze += 1

            # to compute eeg likelihoods, take the next 10 indices of the eeg test data every time in this loop:
            start = test_idx * trials_per_inquiry
            end = (test_idx + 1) * trials_per_inquiry
            eeg_tst_data = preprocessed_test_eeg[:, start:end, :]
            inq_sym = inquiry_symbols_test[start: end]
            eeg_likelihood_ratios = eeg_model.compute_likelihood_ratio(eeg_tst_data, inq_sym, symbol_set)
            unnormalized_log_likelihood_eeg = np.log(eeg_likelihood_ratios)
            eeg_log_likelihoods[test_idx, :] = unnormalized_log_likelihood_eeg
            normalized_posterior_eeg_only = np.exp(
                eeg_log_likelihoods[test_idx, :]) / np.sum(np.exp(eeg_log_likelihoods[test_idx, :]))

            max_like_eeg = np.argmax(normalized_posterior_eeg_only)
            top_competitor_eeg = np.sort(normalized_posterior_eeg_only)[-2]
            posterior_of_true_label_eeg = normalized_posterior_eeg_only[symbol_to_index[gaze_test_labels[test_idx]]]

            target_posteriors_eeg[test_idx, 0] = posterior_of_true_label_eeg
            target_posteriors_eeg[test_idx, 1] = top_competitor_eeg
            if symbol_set[max_like_eeg] == gaze_test_labels[test_idx]:
                counter_eeg += 1

            # Bayesian fusion update and decision making:
            log_unnormalized_posterior = np.log(eeg_likelihood_ratios) + gaze_log_likelihoods[test_idx, :]
            unnormalized_posterior = np.exp(log_unnormalized_posterior)
            denominator = np.sum(unnormalized_posterior)
            posterior = unnormalized_posterior / denominator  # normalized posterior
            log_posterior = np.log(posterior)
            max_posterior = np.argmax(log_posterior)
            top_competitor_fusion = np.sort(log_posterior)[-2]
            posterior_of_true_label_fusion = posterior[symbol_to_index[gaze_test_labels[test_idx]]]

            target_posteriors_fusion[test_idx, 0] = posterior_of_true_label_fusion
            target_posteriors_fusion[test_idx, 1] = top_competitor_fusion
            if symbol_set[max_posterior] == gaze_test_labels[test_idx]:
                counter_fusion += 1

            # stop if posterior has nan values:
            if posterior.any() == np.nan:
                break

        eeg_acc_in_iteration = float("{:.3f}".format(counter_eeg / counter_test_samples))
        gaze_acc_in_iteration = float("{:.3f}".format(counter_gaze / counter_test_samples))
        fusion_acc_in_iteration = float("{:.3f}".format(counter_fusion / counter_test_samples))
        eeg_acc.append(eeg_acc_in_iteration)
        gaze_acc.append(gaze_acc_in_iteration)
        fusion_acc.append(fusion_acc_in_iteration)

    progress_bar.close()

    return eeg_acc, gaze_acc, fusion_acc
