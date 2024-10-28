import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Visualize the results by accessing the data from the csv file:
data_folder = "/Users/basak/Documents/BciPy/data/Multimodal_Subjects/S001/S001_Matrix_Calibration_Mon_15_May_2023_10hr39min42sec_-0400"
df = pd.read_csv(f"{data_folder}/results.csv")
eeg_acc = df['EEG']
gaze_acc = df['Gaze']
fusion_acc = df['Fusion']

plt.bar(['EEG', 'Gaze', 'Fusion'], [np.mean(eeg_acc), np.mean(gaze_acc), np.mean(fusion_acc)])
plt.title(f"Average accuracy over N={len(eeg_acc)} iterations")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 1)
plt.show()

# Plot all accuracies over all iterations, in decreasing order wrt the fusion accuracy:
plt.bar(fusion_acc.index, fusion_acc, label='Fusion')
plt.bar(eeg_acc.index, eeg_acc, label='EEG')
plt.bar(gaze_acc.index, gaze_acc, label='Gaze')
plt.legend()
plt.title("Accuracy over all iterations")
plt.xlabel("Iteration")
plt.ylabel("Test Accuracy (%)")
plt.show()
breakpoint()



