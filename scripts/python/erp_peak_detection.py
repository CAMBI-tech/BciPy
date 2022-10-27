from matplotlib.pyplot import annotate
import mne
from numpy import block

def determine_latency(
        epochs,
        conditions,
        raw,
        semi_automatic=True,
        channels=None,
        duration=0.8,
        label='P300',
        mode='pos',
        tmin=0.2,
        tmax=0.8):
    """Determine the latency an ERP response.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    conditions : list
        The conditions to process for amplitude differences.
    raw : mne.io.RawArray
        The raw data the epochs were extracted from.
    semi_automatic : bool
        If True, the user will be prompted to select the latency of the ERP response.
    channels : list
        The channels to include in the peak detection analysis.
    duration : float
        The duration of the ERP response.
    label : str
        The label to use for the ERP response annotation. e.g. 'P300'
    mode : str
        The mode to use for peak detection. 'pos' or 'neg'.
    tmin : float
        The minimum time in seconds to include in the peak detection analysis.
    tmax : float
        The maximum time in seconds to include in the peak detection analysis.


    Returns
    -------
    latencies: list
        The latency of the ERP response. Each condition is a key in the dictionary and the value is the latency for that condition.
    """
    latencies = []
    for con in conditions:
        average_per_condition = epochs[con].average() 
        ch_name, latency = average_per_condition.get_peak(
            mode=mode, tmin=tmin, tmax=tmax, picks=channels)

        print(f'Peak latency for condition {con} is {latency} in channel {ch_name}')
        
        if semi_automatic:
            # In order to get the correct units etc for the annotation, we need to set and retrieve them from the raw data
            # https://mne.tools/dev/generated/mne.Epochs.html#mne.Epochs.set_annotations
            raw.set_annotations(mne.Annotations(onset=latency, duration=duration, description=label))
            annotate = raw.annotations
            average_per_condition.set_annotations(annotate)
            average_per_condition.plot(block=True)

            latency = average_per_condition.get_annotations_per_epoch()[0][0]['onset']
        
        latencies.append(latency)

    return latencies

def semi_automatic_peak_detection(epochs, conditions, label, raw, duration=0.8, tmin=0.2, tmax=0.8, mode='pos', channels=None):
    """Semi-automatic peak detection.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    conditions : list
        The conditions to process for amplitude differences.

    Returns
    -------

    latecny : dict
        The latency of the ERP response. Each condition is a key in the dictionary and the value is the latency for that condition.
    """
    return determine_latency(
        epochs,
        conditions,
        raw,
        semi_automatic=True,
        label=label,
        duration=duration,
        channels=channels,
        tmin=tmin,
        tmax=tmax,
        mode=mode)

if __name__ == '__main__':
    p300_detection = False
    n200_detection = False


    # TODO: Load the data
    epochs = None
    raw = None
    participants = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    # Define the conditions
    conditions = ['nontarget', 'target']

    # Define the channels to use for peak detection
    channels = ['Pz', 'POz', 'Oz', 'PO3', 'PO4', 'O1', 'O2']

    for _ in participants:
        if p300_detection:
            # Define the ERP response to detect
            label = 'P300'

            # Define the ERP response duration
            duration = 0.8

            # Define the ERP response mode
            mode = 'pos'

            # Define the ERP response time window
            tmin = 0.2
            tmax = 0.8

            # Semi-automatic peak detection
            p300_latencies = semi_automatic_peak_detection(epochs, conditions, label, raw, duration=duration, tmin=tmin, tmax=tmax, mode=mode, channels=channels)
        

        if n200_detection:
            # Define the ERP response to detect
            label = 'N200'

            # Define the ERP response duration
            duration = 0.8

            # Define the ERP response mode
            mode = 'neg'

            # Define the ERP response time window
            tmin = 0.1
            tmax = 0.3

            # Semi-automatic peak detection
            n200_latencies = semi_automatic_peak_detection(epochs, conditions, label, raw, duration=duration, tmin=tmin, tmax=tmax, mode=mode, channels=channels)
    

