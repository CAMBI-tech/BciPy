# -*- coding: utf-8 -*-
from acquisition.sig_pro import sig_pro
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

"""
Test of filter on real data 
"""

# reading previous filter from .mat file
Data = loadmat('sample_dat.mat')
EEG_Data = np.transpose(Data['x'])
fs = 256  # downsampling factor
k=1 # downsampling factor

# New filter: testing filter on real data
y = sigPro(EEG_Data, fs = fs, k = 1, channels = 20)

# Old filter: testing filter on real data
filt = loadmat('inputFilterCoef.mat')
groupDelay = filt['frontendFilter']['groupDelay'][0,0][0][0]
filterNum = filt['frontendFilter']['Num'][0,0][0] # Den = 1
# Convolution per channel
temp = np.convolve(EEG_Data[0][:], filterNum)
# Filter off-set compensation
temp = temp[groupDelay-1:];
# Downsampling
Y = temp[::k]

plt.figure(1)
plt.plot(EEG_Data[0][0:200],'b')
plt.plot(y[0][0:200],'r')
plt.plot(Y[0:200],'g')
plt.title('EEG sample data before and after filtering')
plt.legend(('Before filtering', 'After filtering_new filter','After filtering_old filter'))
plt.show()