from numpy import ndarray
import numpy as np
from bcipy.helpers.symbols import alphabet


def calculate_eeg_gaze_fusion_acc(
        eeg_model,
        gaze_model,
        test_eeg_data,
        test_gaze_data,
        symbol_set,
        parameters,
        data_folder):
    """
    Preprocess the EEG and gaze data. Calculate the accuracy of the fusion of EEG and Gaze models.

    Args:
        eeg_model: EEG model
        gaze_model: Gaze model
        test_eeg_data: EEG data that the model is tested on (not trained on)
        test_gaze_data: Gaze data that the model is tested on (not trained on)
    Returns:
        float: accuracy of the fusion TODO: before releasing 2.0 figure out a way to standardize the return type
    """
    # Preprocess the EEG data:
    symbol_set = alphabet()

    # selection length is the length of eeg or gaze data, whichever is smaller:
    eeg_inquiries, eeg_inquiry_labels, eeg_inquiry_timing = eeg_model.reshaper(
        trial_targetness_label=trigger_targetness,
        timing_info=corrected_trigger_timing,
        eeg_data=test_eeg_data,
        sample_rate=eeg_sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=eeg_channel_map,
        poststimulus_length=window_length,
        prestimulus_length=prestim_length,
        transformation_buffer=buffer,
    )

    eeg_inquiry_timing = np.array(eeg_inquiry_timing)
    train_eeg_inquiry_timing = eeg_inquiry_timing[train_indices]
    test_eeg_inquiry_timing = eeg_inquiry_timing[test_indices]
    inquiry_symbols_test = np.array([])
    for t_i in test_indices:
        inquiry_symbols_test = np.append(inquiry_symbols_test, inquiry_symbols[t_i * 10:(t_i + 1) * 10])
    inquiry_symbols_test = inquiry_symbols_test.tolist()

    # Now extract the inquiries from trials for eeg model fitting:
    preprocessed_train_eeg = eeg_model.reshaper.extract_trials(
        train_data_eeg, trial_duration_samples, train_eeg_inquiry_timing)
    preprocessed_test_eeg = eeg_model.reshaper.extract_trials(
        test_data_eeg, trial_duration_samples, test_eeg_inquiry_timing)

    # train and save the eeg model a pkl file
    log.info("Training model. This will take some time...")
    # Flatten the labels (0=nontarget/1=target) prior to model fitting
    erp_train_labels = eeg_inquiry_labels[train_indices].flatten().tolist()
    # erp_test_labels = eeg_inquiry_labels[test_indices].flatten().tolist()
    eeg_model.fit(preprocessed_train_eeg, erp_train_labels)
    eeg_model.metadata = SignalModelMetadata(device_spec=device_spec_eeg,
                                             transform=default_transform)
    log.info(f"Training complete [AUC={eeg_model.auc:0.4f}]. Saving data...")
    # save_model(eeg_model, Path(data_folder, f"model_{eeg_model.auc:0.4f}.pkl"))
    preferences.signal_model_directory = data_folder

    # extract train and test indices for gaze data:
    centralized_gaze_data_train = centralized_gaze_data[train_indices]
    gaze_train_labels = np.array([target_symbols[i] for i in train_indices])
    gaze_data_test = preprocessed_gaze_data[test_indices]         # test set is NOT centralized
    gaze_test_labels = np.array([target_symbols[i] for i in test_indices])
    # generate a tuple that matches the index of the symbol with the symbol itself:
    symbol_to_index = {symbol: i for i, symbol in enumerate(symbol_set)}

    # train and save the gaze model as a pkl file:
    reshaped_data = centralized_gaze_data_train.reshape((len(centralized_gaze_data_train), 720))
    units = 1e4
    reshaped_data *= units
    cov_matrix = np.cov(reshaped_data, rowvar=False)
    time_horizon = 9

    for eye_coord_0 in range(4):
        for eye_coord_1 in range(4):
            for time_0 in range(180):
                for time_1 in range(180):
                    l_ind = eye_coord_0 * 180 + time_0
                    m_ind = eye_coord_1 * 180 + time_1
                    if np.abs(time_1 - time_0) > time_horizon:
                        cov_matrix[l_ind, m_ind] = 0

    reshaped_mean = np.mean(reshaped_data, axis=0)

    # eps = 5e+3
    eps = 0
    regularized_cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * eps
    try:
        inv_cov_matrix = np.linalg.inv(regularized_cov_matrix)
    except BaseException:
        print("Singular matrix, using pseudo-inverse instead")
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

    # Initialize the accuracy lists:
    eeg_acc = []
    gaze_acc = []
    fusion_acc = []

    inq_length = parameters.get('stim_length', 10)
    counter_gaze = 0
    counter_eeg = 0
    counter_fusion = 0
    # TODO pull parameters that are needed from the parameters dictionary or ask it to be passed by the caller!
    for test_idx, test_data in enumerate(test_gaze_data):

        numerator_gaze_list = []
        diff_list = []
        for idx, sym in enumerate(symbol_set):
            central_data = gaze_model.substract_mean(test_data, time_average_per_symbol[sym])
            flattened_data = central_data.reshape((720,))
            flattened_data *= units
            diff = flattened_data - reshaped_mean
            diff_list.append(diff)
            numerator = -np.dot(diff.T, np.dot(inv_cov_matrix, diff)) / 2  # !!!!!!
            numerator_gaze_list.append(numerator)
            unnormalized_log_likelihood_gaze = numerator - denominator_gaze
            gaze_log_likelihoods[test_idx, idx] = unnormalized_log_likelihood_gaze

        normalized_posterior_gaze_only = np.exp(
            gaze_log_likelihoods[test_idx, :]) / np.sum(np.exp(gaze_log_likelihoods[test_idx, :]))
        # Find the max likelihood:
        max_like_gaze = np.argmax(normalized_posterior_gaze_only)

        posterior_of_true_label_gaze = normalized_posterior_gaze_only[symbol_to_index[gaze_test_labels[test_idx]]]
        top_competitor_gaze = np.sort(normalized_posterior_gaze_only)[-2]
        target_posteriors_gaze[test_idx, 0] = posterior_of_true_label_gaze
        target_posteriors_gaze[test_idx, 1] = top_competitor_gaze
        # Check if it's the same as the target
        if symbol_set[max_like_gaze] == gaze_test_labels[test_idx]:
            counter_gaze += 1

        # EEG model:
        start = test_idx * inq_length
        end = (test_idx + 1) * inq_length
        eeg_tst_data = test_eeg_data[:, start:end, :]
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
            raise ValueError("Posterior has nan values")

    # TODO: cleanup! There will only be one loop now
    # print accuracy with only 3 decimals:
    eeg_acc_in_iteration = float("{:.3f}".format(counter_eeg / len(test_indices)))
    gaze_acc_in_iteration = float("{:.3f}".format(counter_gaze / len(test_indices)))
    fusion_acc_in_iteration = float("{:.3f}".format(counter_fusion / len(test_indices)))
    eeg_acc.append(eeg_acc_in_iteration)
    gaze_acc.append(gaze_acc_in_iteration)
    fusion_acc.append(fusion_acc_in_iteration)
    print(f"Iteration: {iter}")
    print(f"# Train samples: {len(train_indices)}")
    print(f"# Test samples: {len(test_indices)}")
    print(f"Gaze accuracy: {gaze_acc_in_iteration}, \nEeg acc: {eeg_acc_in_iteration}")
    print(f"Fusion acc: {fusion_acc_in_iteration}")

    # Save it to a csv file:
    results = pd.DataFrame({'EEG': eeg_acc, 'Gaze': gaze_acc, 'Fusion': fusion_acc})
    results.to_csv(f"{data_folder}/results.csv")

    print(f"Average EEG accuracy: {np.mean(eeg_acc)}")
    print(f"Average Gaze accuracy: {np.mean(gaze_acc)}")
    print(f"Average Fusion accuracy: {np.mean(fusion_acc)}")

    return 'TODO: fusion Accuracy'

# if __name__ == "__main__":
#     calculate_eeg_gaze_fusion_acc()
