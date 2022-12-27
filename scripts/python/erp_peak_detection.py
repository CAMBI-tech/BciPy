import mne
mne.set_log_level('WARNING')
from pathlib import Path
from bcipy.helpers.load import (
    load_experimental_data,
)

def determine_latency(
        epochs,
        semi_automatic=True,
        prelabel=False,
        label='P300',
        mode='pos',
        tmin=0.2,
        tmax=0.8):
    """Determine the latency an ERP response.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    semi_automatic : bool
        If True, the user will be prompted to select the latency of the ERP response.
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

    average_per_condition = epochs.average()
    
    if not prelabel:
        ch_name, latency = average_per_condition.get_peak(
            mode=mode, tmin=tmin, tmax=tmax)
        print(f'{label} latency: {latency} in channel {ch_name} detected automatically. \n'
           'Proceeding to semi-automatic peak detection. \n')
    else:
        latency = prelabel
        print(f'{label} latency: {latency} detected by prelabel. \n'
           'Proceeding to semi-automatic peak detection. \n')

    if semi_automatic:
        latency = get_latency(average_per_condition, latency, label)
        print(f'{label} latency: {latency}')
    return latency

def get_latency(average, latency, condition):
    get_input = True
    while get_input:
        fig = average.plot(show=True, window_title=f'Average {condition}, Estimated: {latency}')
        user_input = input(f'Is the latency=[{latency}] correct? (y/n): ')
        if user_input == 'y':
            get_input = False
        elif user_input == 'n':
            latency = input('Enter the latency in seconds: ')
            get_input = True
            condition += '-CONFIRM'
        else:
            print('Invalid input. Please enter y or n.')
        try:
            return float(latency)
        except ValueError:
            print('Invalid input. Please enter a number for latency.')
            get_input = True

def semi_automatic_peak_detection(epochs, label, prelabel=None, tmin=0.2, tmax=0.8, mode='pos'):
    """Semi-automatic peak detection.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.

    Returns
    -------

    latecny : dict
        The latency of the ERP response. Each condition is a key in the dictionary and the value is the latency for that condition.
    """
    return determine_latency(
        epochs,
        prelabel=prelabel,
        semi_automatic=True,
        label=label,
        tmin=tmin,
        tmax=tmax,
        mode=mode)

def detection(epochs, condition, prelabel=None):
    """Peak detection.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    condition : str
        The condition to process.
    prelabel : str
        The label to use for the ERP response annotations. First must be p300, last must be n200.

    Returns
    -------

    n2 latency, p3 latency, averaged epoches : tuple
        The latency of the ERP response. Each condition is a key in the dictionary and the value is the latency for that condition.
    """

    # P300
    # Define the ERP response to detect
    label = f'P300-{condition}'
    tmin = 0.3
    tmax = 0.8
    # duration = 0.3
    # Define the ERP response mode
    mode = 'pos'
    # Semi-automatic peak detection
    p300_latencies = semi_automatic_peak_detection(
        epochs,
        label,
        prelabel=prelabel[0] if prelabel else None,
        tmin=tmin,
        tmax=tmax,
        mode=mode)

    # N200
    # Define the ERP response to detect
    label = f'N200-{condition}'
    # Define the ERP response mode
    mode = 'neg'
    # Define the ERP response time window
    tmin = 0.2
    tmax = 0.325
    # duration = 0.3

    # Semi-automatic peak detection
    n200_latencies = semi_automatic_peak_detection(
        epochs, 
        label,
        prelabel=prelabel[-1] if prelabel else None,
        tmin=tmin,
        tmax=tmax,
        mode=mode)
    
    return n200_latencies, p300_latencies, epochs.average()


if __name__ == '__main__':
    write_output = True
    target_filename = 'target-epo.fif'
    nontarget_filename = 'nontarget-epo.fif'

    # process the target data first and use those as the first place label for the nontarget data
    path = load_experimental_data()
    
    for session in Path(path).iterdir():
        try:
            print(f'\nProcessing {session} \n')
            # mne_data = mne.io.read_raw_fif(f'{session}/{raw_data_filename}', preload=True)
            target = mne.read_epochs(f'{session}/{target_filename}', preload=True)
            n2_target, p3_target, target_average = detection(target, condition='Target')
            nontarget = mne.read_epochs(f'{session}/{nontarget_filename}', preload=True)
            n2_nontarget, p3_nontarget, nontarget_average = detection(
                nontarget,
                condition='NonTarget',
                prelabel=[p3_target, n2_target])

            if write_output:
                print(f'\nWriting output {session}/n2_p3_latencies.txt \n')
                with open(f'{session}/n2_p3_latencies.txt', 'w') as f:
                    f.write(f'Label, Target, Nontarget \n')
                    f.write(f'N2, {n2_target}, {n2_nontarget} \n')
                    f.write(f'P3, {p3_target}, {p3_nontarget} \n')
            
            print('Complete! \n')
            
        except Exception as e:
            print(f'Could not load epochs for session {session}: [{e}]')
            continue

    
