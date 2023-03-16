import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

betts_data_path = '/Users/basak/Documents/BciPy/data/MatrixEyeTracking/BP_001/BP_001_Matrix_Calibration_Wed_08_Mar_2023_12hr39min04sec_-0800/raw_data.csv'
betts_triggers_path = '/Users/basak/Documents/BciPy/data/MatrixEyeTracking/BP_001/BP_001_Matrix_Calibration_Wed_08_Mar_2023_12hr39min04sec_-0800/triggers.txt'

if __name__ == "__main__":
    df = pd.read_csv(betts_data_path)
    df = df.dropna()
    # df = df.reset_index(drop=True)
    X = df[['left_x', 'left_y']].to_numpy()
    timestamp = df['lsl_timestamp'].to_numpy()
    gm = GaussianMixture(n_components=28, random_state=0).fit(X)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    # plt.scatter(X[:, 2], X[:, 3])
    # plt.show()
    print(gm.means_)
    plt.scatter(gm.means_[:,0], gm.means_[:,1])
    plt.show()
    # evaluate the components’ density for each sample
    conditionals = gm.predict_proba(X)
    breakpoint()