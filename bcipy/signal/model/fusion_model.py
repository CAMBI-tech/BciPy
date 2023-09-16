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
    
    def get_scores(self, test_data: np.array):
        '''
        Compute the log-likelihood of each sample.
        Compute the mean and covariance of each mixture component.
        '''
        
        scores = self.model.score_samples(test_data)
        means = self.model.means_
        covs = self.model.covariances_
        
        return scores, means, covs
    
    def predict(self, scores: np.array):
        '''
        Predict the labels for the test data.
        '''
        # Compute over log-likelihood scores
        # Get the argmax of the scores
        
        #return predictions
