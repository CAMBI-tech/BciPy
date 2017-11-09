import numpy as np
import pandas as pd


def read_data_csv(folder, dat_first_row=4, info_end_row=1):
    """ Reads the data (.csv) provided by the data acquisition
        Arg:
            folder(str): file location for the data
            dat_first_row(int): row with channel names
            info_end_row(int): final row related with daq. info
                where first row idx is '0'
        Return:
            raw_dat(ndarray[float]): C x N numpy array with samples
                where C is number of channels N is number of time samples
            channels(list[str]): channels used in DAQ
            stamp_time(ndarray[float]): time stamps for each sample
            type_amp(str): type of the device used for DAQ
            fs(int): sampling frequency
    """
    dat_file = pd.read_csv(folder, skiprows=dat_first_row)

    cols = list(dat_file.axes[1])
    channels = cols[2:len(cols)]

    temp = np.array(dat_file)
    stamp_time = temp[:, 0]
    raw_dat = temp[:, 1:temp.shape[1]].transpose()

    dat_file_2 = pd.read_csv(folder, nrows=info_end_row)
    type_amp = list(dat_file_2.axes[1])[1]
    fs = np.array(dat_file_2)[0][1]

    return raw_dat, stamp_time, channels, type_amp, fs


def test_read_data():
    # Might cause path issues please be aware!
    location = 'rawdata.csv'
    raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(location)
    print('Channels:{}, AMP:{}, fs:{}'.format(channels, type_amp, fs))

    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pylab as plt

    plt.plot(stamp_time, raw_dat.transpose(), '*')
    plt.show()

    return 0


def main():
    test_read_data()

    return 0


if __name__ == "__main__":
    main()
