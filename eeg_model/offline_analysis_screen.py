""" Offline Analysis Screen. \
Generates the information figure following
the offlineAnalysis.
The figure has multiple tabs containing the average ERP plots
 """

# import
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load data
fol = loadmat('dummy_dat.mat')
y = fol['trialTargetness']
y = np.squeeze(y)
x = fol['trialData']

# convert from V to microVolts the data
x = x * np.power(10, 6)

classes = np.unique(y)
means = [np.mean(x[:, :, np.where(y == i)], 3) for i in classes]

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

count = 1
while count < means[1].shape[1]:
    ax1.plot(means[0][:, count])
    ax1.plot(means[0][:, count])
    ax2.plot(means[1][:, count])
    ax2.plot(means[1][:, count])
    count += 1

# Set common labels
fig.text(0.5, 0.04, 'time Samples [n]', ha='center', va='center')
fig.text(0.06, 0.5, '$\mu V$', ha='center', va='center', rotation='vertical')

ax1.set_title(
    'Mean distractor ERP (averaged over trials in the calibration data')
ax2.set_title('Mean target ERP (averaged over trials in the calibration data)')

plt.show()
