def determine_p300_amplitude(epochs, conditions, channels=None):
    """Determine the amplitude of the P300 response.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    conditions : list
        The conditions to process for amplitude differences.
    channels : list
        The channels to include in the peak detection analysis.

    Returns
    -------
    p300_amplitude : float
        The amplitude of the P300 response.
    p300_latency : float
        The latency of the P300 response.
    p300_max : float
        The maximum voltage channel location of the P300 response.
    """
    p300_amplitudes = []
    p300_latency = []
    p300_maximal_location = []
    for con in conditions:
        average_per_condition = epochs[con].average()
        ch_name, latency, amplitude = average_per_condition.get_peak(
            mode='pos', return_amplitude=True, tmin=0.2, tmax=0.5)
        p300_amplitudes.append(amplitude)
        p300_latency.append(latency)
        p300_maximal_location.append(ch_name)
    return p300_amplitudes, p300_latency, p300_maximal_location

def semi_automatic_peak_detection(epochs, conditions):
    """Semi-automatic peak detection.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    conditions : list
        The conditions to process for amplitude differences.

    Returns
    -------
    p300_amplitude : float
        The amplitude of the P300 response.
    p300_latency : float
        The latency of the P300 response.
    p300_max : float
        The maximum voltage channel location of the P300 response.
    """
    p300_amplitudes, p300_latency, p300_maximal_location = determine_p300_amplitude(epochs, conditions)
    for con, amp, lat, loc in zip(conditions, p300_amplitudes, p300_latency, p300_maximal_location):
        print(f'Condition: {con}, Amplitude: {amp}, Latency: {lat}, Location: {loc}')
        # TODO: add interactive plot here by making an annotation and ploting agains the epoch average for each condition
    return None