# -*- coding: utf-8 -*-
from acquisition.sig_pro import sig_pro
import numpy as np
from scipy.io import loadmat

"""
test the RSVP_keyboard previous EEG filter
"""

fs = 256 # (hertz) sampling frequency
Duration = 5 # (seconds)  test duration
t = np.arange(0, Duration, 1./fs) # time vector
k = 1 #downsampling factor


# sinusoid in different frequencies 
x0 = np.cos(2*np.pi*2*t)
x1 = np.cos(2*np.pi*10*t)
x2 = np.cos(2*np.pi*20*t)
x3 = np.cos(2*np.pi*30*t)
x4 = np.cos(2*np.pi*40*t)
x6 = 1
x7 = np.cos(2*np.pi*1*t)
x8 = np.cos(2*np.pi*48*t)
x9 = np.cos(2*np.pi*70*t)
x10 = np.cos(2*np.pi*100*t)

xpassband = x0 + x1 + x2 + x3 + x4
xstopband = x6 + x7 + x8 + x9 + x10

#one channels
x = np.zeros((1,xpassband.size))
x[0][:] = xpassband + xstopband


# reading previous filter from .mat file
filt = loadmat('inputFilterCoef.mat')
groupDelay = filt['frontendFilter']['groupDelay'][0,0][0][0]
filterNum = filt['frontendFilter']['Num'][0,0][0] # Den = 1
# Convolution per channel
temp = np.convolve(x[0][:], filterNum)
# Filter off-set compensation
temp = temp[groupDelay-1:];
# Downsampling
output = temp[::k]

input_size = x.size
output_size = output.size
MSE = np.sum((xpassband - output[:xpassband.size])**2.)/xpassband.size
Normalized_MSE = MSE/np.sum(x[0][:])*100

print 'MSE: {}'.format(MSE)
print 'MSE normalized over input signal power (percentage): {}%'.format(Normalized_MSE)