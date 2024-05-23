import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

all_accuracies = pd.read_csv('all_data_ind.csv')

# Plot box plots for each column
plt.figure()
all_accuracies.mask(all_accuracies == 0).plot.box()
plt.title('Symbol Selection Accuracies, Individual Gaze Model')
plt.ylabel('Accuracy')
plt.xlabel('Symbol')
plt.ylim(0, 100)
plt.show()

subj_ids = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

all_results = pd.read_csv('results.csv')
# Plot each column as a bar plot
plt.figure()
all_results.plot.bar()
plt.title('Overall Accuracy (%)')
plt.ylabel('Accuracy')
plt.xlabel('Subject ID')
# add x axis labels
plt.xticks(all_results.index, subj_ids, rotation=0)
plt.legend(loc='lower left')
plt.show()




