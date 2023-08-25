import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from bcipy.helpers.stimuli import GazeReshaper

class GazeModel():
    reshaper = GazeReshaper()

    def __init__(self, num_components=3):
        self.num_components = num_components   # number of gaussians to fit
    
    def fit(self, train_data: np.array):
        model = GaussianMixture(n_components=2, random_state=0, init_params='kmeans')
        model.fit(train_data)
        self.model = model

        return self
    
    def predict(self, test_data: np.array):
        predictions = self.model.predict_proba(test_data)
        means = self.model.means_
        covs = self.model.covariances_
        
        return predictions, means, covs



# if __name__ == "__main__":
#     # Dummy model test
#     letterdata = np.load('/Users/basak/Documents/BciPy/S001_A.npz')
#     left_eye =  np.vstack((letterdata['lx'], letterdata['ly'])).T 
#     right_eye = np.vstack((letterdata['rx'], letterdata['ry'])).T

#     # Train test split the data
#     test_size_right = int(len(right_eye) * 0.2)
#     train_size_right = len(right_eye) - test_size_right
#     train_right_eye = right_eye[:train_size_right]
#     test_right_eye = right_eye[train_size_right:]

#     test_size_left = int(len(left_eye) * 0.2)
#     train_size_left = len(left_eye) - test_size_left
#     train_left_eye = left_eye[:train_size_left]
#     test_left_eye = left_eye[train_size_left:]

#     gm = GaussianMixture(n_components=2, random_state=0, init_params='kmeans').fit(train_right_eye)
#     predictions = gm.predict_proba(test_right_eye)
#     print(predictions)
#     plt.scatter(left_eye[:, 0], left_eye[:, 1], c='r', s=1)
#     plt.scatter(right_eye[:, 0], right_eye[:, 1], c='b', s=1)
#     means = gm.means_
#     plt.scatter(means[:,0], means[:,1], c='black', s=20, marker='^')
#     plt.show()
#     breakpoint()
