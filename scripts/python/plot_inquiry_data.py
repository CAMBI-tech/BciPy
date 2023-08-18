from bcipy.helpers.raw_data import load
from bcipy.config import STATIC_IMAGES_PATH

DIPSIZE = (1707, 1067)
IMG_PATH = f'{STATIC_IMAGES_PATH}/main/matrix.png'
TOBII_FILENAME = 'eyetracker_data_tobii-p0.csv'

def load_eye_tracking_data(path):
    data = load(f'{path}/{TOBII_FILENAME}')

    left_eye_channel_map = [0,0,1,1,1,0,0,0,0]
    left_eye_data, _, _ = data.by_channel_map(left_eye_channel_map)
    left_eye_x = left_eye_data[0]
    left_eye_y = left_eye_data[1]
    left_pupil = left_eye_data[2]

    right_eye_channel_map = [0,0,0,0,0,1,1,1,0]
    right_eye_data, _, _ = data.by_channel_map(right_eye_channel_map)
    right_eye_x = right_eye_data[0]
    right_eye_y = right_eye_data[1]
    right_pupil = right_eye_data[2]
    return left_eye_x, left_eye_y, left_pupil, right_eye_x, right_eye_y, right_pupil


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from bcipy.helpers.load import load_experimental_data
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
    lx, ly, lp, rx, ry, rp = data
   

    dpi = 100

    # load the image
    img = plt.imread(IMG_PATH)
    # img = np.flipud(img)

    w, h = len(img[0]), len(img)

    # resize the image to fit the display
    # img = np.resize(img, (DIPSIZE[1], DIPSIZE[0], 4))
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 1, 0, 1])
    # ax.scatter(lx, ly, c='r', s=1)
    # ax.scatter(rx, ry, c='b', s=1)
    # transform the eye data to fit the display. remove > 1 values < 0 values and flip the y axis
    lx = np.clip(lx, 0, 1)
    ly = np.clip(ly, 0, 1)
    rx = np.clip(rx, 0, 1)
    ry = np.clip(ry, 0, 1)
    ly = 1 - ly
    ry = 1 - ry

    # plot the eye data
    # ax.plot(lx, ly, c='r', linewidth=1)
    # ax.plot(rx, ry, c='b', linewidth=1)

    # # remove nan values
    lx = lx[~np.isnan(lx)]
    ly = ly[~np.isnan(ly)]
    rx = rx[~np.isnan(rx)]
    ry = ry[~np.isnan(ry)]

    # determine kde of the eye data
    # ax = sns.kdeplot(rx, ry, cmap="mako", fill=False, thresh=0.05, levels=10)


    ax.scatter(lx, ly, c='r', s=1)
    ax.scatter(rx, ry, c='b', s=1)

    plt.show()

    breakpoint()