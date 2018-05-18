from helpers.load import read_data_csv, load_experimental_data
from signal_processing.sig_pro import sig_pro
from signal_model.mach_learning.train_model import train_pca_rda_kde_model
from signal_model.mach_learning.trial_reshaper import trial_reshaper
from helpers.data_viz import generate_offline_analysis_screen
from helpers.triggers import trigger_decoder
import pickle

# Channels relevant for analysis, for each supported device.
# TODO: Is there a better place to put this information?
ANALYSIS_CHANNELS = {
    'DSI': ["P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "A1", "Fp1", "Fp2",
            "T3", "T5", "O1", "O2", "F7", "F8", "A2", "T6", "T4"],
    'g.USBamp-2': ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8",
                   "Ch9", "Ch10", "Ch11", "Ch12", "Ch13", "Ch14", "Ch15",
                   "Ch16"]
}


def analysis_channels(channels, device_name):
    """
    Parameters:
    ----------
        channels(list(str)): list of channel names from the raw_data
            (excluding the timestamp)
        device_name(str): daq_type from the raw_data file.
    Returns:
    --------
        A binary list indicating which channels should be used for analysis.
        If i'th element is 0, i'th channel in filtered_eeg is removed.
    """
    relevant_channels = ANALYSIS_CHANNELS.get(device_name)
    if not relevant_channels:
        raise Exception("Analysis channels for the given device not found.")
    return [int(ch in relevant_channels) for ch in channels]


def offline_analysis(data_folder=None):
    """ Gets calibration data and trains the model in an offline fashion.
        pickle dumps the model into a .pkl folder
        Args:
            data_folder(str): folder of the data
                save all information and load all from this folder

        Duty cycle
        - reads data and information from a .csv calibration file
        - reads trigger information from a .txt trigger file
        - filters data
        - reshapes and labels the data for the training procedure
        - fits the model to the data
            - uses cross validation to select parameters
            - based on the parameters, trains system using all the data
        - pickle dumps model into .pkl file
        - generates and saves offline analysis screen
        """

    if not data_folder:
        data_folder = load_experimental_data()

    mode = 'calibration'

    # TODO: raw_data.csv can be configured; this value should be read in from
    # the params.
    raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
        data_folder + '/raw_data.csv')

    print(channels)
    print(type_amp)
    # TODO: Read from parameters
    ds_rate = 2
    dat = sig_pro(raw_dat, fs=fs, k=ds_rate)

    # Process triggers.txt
    s_i, t_t_i, t_i, offset = trigger_decoder(mode=mode,
                                              trigger_loc=data_folder + '/triggers.txt')

    # Channel map can be checked from raw_data.csv file.
    # read_data_csv already removes the timespamp column.
    channel_map = analysis_channels(channels, type_amp)

    x, y, num_seq, _ = trial_reshaper(t_t_i, t_i, dat,
                                      mode=mode, fs=fs, k=ds_rate,
                                      offset=offset,
                                      channel_map=channel_map)

    model = train_pca_rda_kde_model(x, y, k_folds=10)

    print('Saving offline analysis plots!')
    generate_offline_analysis_screen(x, y, model, data_folder)

    print('Saving the model!')
    with open(data_folder + '/model.pkl', 'wb') as output:
        pickle.dump(model, output)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default=None)
    args = parser.parse_args()

    offline_analysis(args.data_folder)
