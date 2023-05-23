import os
from bcipy.helpers.raw_data import load

DIPSIZE = (1707, 1067)
TOBII_FILENAME = 'eyetracker_data_tobii-p0.csv'
# IMG_PATH = os.path.abspath(os.path.dirname(__file__))+ '/matrix_grid.png'

def load_eye_tracking_data(path):
    data = load(f'{path}/{TOBII_FILENAME}')
    return data

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from bcipy.helpers.load import load_experimental_data
    from bcipy.helpers.visualization import visualize_gaze
    import matplotlib.pyplot as plt
    from matplotlib import image
    import numpy as np
    import seaborn as sns
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the directory with raw_data to be converted',
        required=False)
    args = parser.parse_args()

    # if no path is provided, prompt for one using a GUI
    path = args.path
    if not path:
        path = load_experimental_data()

    data = load_eye_tracking_data(path)

    # Plot the eye data, set save_path to None to not save the plot. You can have both plots at the same time.
    visualize_gaze(data, show=True, save_path=path, heatmap=False, raw_plot=True)
