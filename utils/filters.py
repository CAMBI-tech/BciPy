# TODO: THIS FILE HAS toggleNumpy which is redundant or we can remove the other

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def test_filter(num_taps, bands, desired, grid_density):
    num_taps = num_taps[0]
    bands = bands[0]
    desired = desired[0]
    bp_flt = signal.remez(num_taps, bands, desired, weight=None, Hz=1,
                          type='bandpass', maxiter=25,
                          grid_density=grid_density)

    return bp_flt


# num_taps = 153
# fs = 256
#
# bands = np.array([0, 0.07, 2.15, 40, 44, fs / 2]) / fs
# # bands = [0, 0.1, 0.2, 0.4, 0.45, 0.5] / fs
# desired = [0, 1, 0]
#
# bp_flt = signal.remez(num_taps, bands, desired, weight=None, Hz=1,
#                       type='bandpass', maxiter=25 * 200, grid_density=20)
#
# freq, response = signal.freqz(bp_flt)
#
# fig = plt.figure()
# plt.title('Digital filter frequency response')
# ax1 = fig.add_subplot(111)
# plt.plot(freq, 20 * np.log10(abs(response)), 'b')
# plt.ylabel('Amplitude [dB]', color='b')
# plt.xlabel('Frequency [rad/sample]')
#
# ax2 = ax1.twinx()
# angles = np.unwrap(np.angle(response))
# plt.plot(freq, angles, 'g')
# plt.ylabel('Angle (radians)', color='g')
# plt.grid()
# plt.axis('tight')
# plt.show()
#
# w, gd = signal.group_delay((np.array([1]), bp_flt), freq)
# plt.title('Digital filter group delay')
# plt.plot(w, gd)
# plt.ylabel('Group delay [samples]')
# plt.xlabel('Frequency [rad/sample]')
# plt.show()
#
# dat = sio.loadmat('C:\Users\Aziz\Desktop\GIT\TMbci\utils\sample_dat.mat')
# x = dat['x']
# x = x * np.power(10, 6)
#
# y = []
# for idx in range(x.shape[1]):
#     y.append(np.convolve(x[:, idx], bp_flt))
# y = np.asarray(y)
# y = y.transpose()
#
# plt.subplot(2, 1, 1)
# plt.plot(x[:, 1:6])
# plt.title('Original data')
# plt.subplot(2, 1, 2)
# plt.plot(y[:, 1:6])
# plt.title('Filtered data')
# plt.show()
