import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

all_accuracies = pd.read_csv('all_data_cent.csv')

# Plot box plots for each column
plt.figure()
all_accuracies.mask(all_accuracies == 0).plot.box()
plt.title('Symbol selection accuracies, Centralized Gaze Model')
plt.ylabel('Accuracy')
plt.xlabel('Symbol')
plt.show()





