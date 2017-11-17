from acquisition.sig_pro.sig_pro import sig_pro
import numpy as np
import matplotlib.pyplot as plt


'''Parameters entered for test here: '''
dcLevel = .1

# passband freq should be chosen w.r.t. filter specification, changes for filter to filter, defined in eegProcessor.py
fcos0 = 2
fcos1 = 10
fcos2 = 20
fcos3 = 30
fcos4 = 40

fs = 256
# fs = 300
# fs = 1024
Duration = 5 # in seconds for test duration
'''End of parameters'''

t = np.arange(0, Duration-1./fs, 1./fs) # time vector

# Cosines with different frequencies
x0 = np.cos(2*np.pi*fcos0*t)
x1 = np.cos(2*np.pi*fcos1*t)
x2 = np.cos(2*np.pi*fcos2*t)
x3 = np.cos(2*np.pi*fcos3*t)
x4 = np.cos(2*np.pi*fcos4*t)

x6 = dcLevel
x7 = np.cos(2*np.pi*1*t)
x8 = np.cos(2*np.pi*48*t)
x9 = np.cos(2*np.pi*70*t)
x10 = np.cos(2*np.pi*100*t)

xpassband = x0 + x1 + x2 + x3 + x4
xstopband = x6 + x7 + x8 + x9 + x10

# Two channels
x = np.zeros((2,xpassband.size))
x[0][:] = xpassband + xstopband
x[1][:] = xpassband + xstopband

y = sig_pro(x, fs = fs, k = 1)

MSE_perSample = np.sum((xpassband - y[0][:xpassband.size])**2.)/xpassband.size
MSE_perSample_norm = MSE_perSample/np.sum(x[0][:]**2) # Decide on which mse to use. Talk with Bahar.

print 'MSE per sample: {} -> {}%'.format(MSE_perSample, MSE_perSample/5.*100)
print 'MSE normalized: {}'.format(MSE_perSample_norm*100)


plt.figure(1)
plt.plot(t, xpassband, t,y[0][:xpassband.size])
plt.title('Sum of Pass-band Cosines')
plt.legend(('Pass-band signal', 'Filtered Signal'))
plt.show()
