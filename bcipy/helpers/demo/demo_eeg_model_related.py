import numpy as np
from bcipy.helpers.eeg_model_related import CopyPhraseWrapper
from bcipy.signal_model.mach_learning.train_model import train_pca_rda_kde_model
from bcipy.helpers.bci_task_related import alphabet

channel_map = [0] + [1] * 16 + [0, 0, 1, 1, 0, 1, 1, 1, 0]
dim_x = 5
num_ch = len(channel_map)

# Make up some distributions that data come from
mean_pos = .5
var_pos = .5
mean_neg = 0
var_neg = .5


def demo_copy_phrase_wrapper():
    # We need to train a dummy model
    num_x_p = 100
    num_x_n = 900

    x_p = mean_pos + var_pos * np.random.randn(num_ch, num_x_p, dim_x)
    x_n = mean_neg + var_neg * np.random.randn(num_ch, num_x_n, dim_x)
    y_p = [1] * num_x_p
    y_n = [0] * num_x_n

    train_x = np.concatenate((x_p, x_n), 1)
    train_y = np.concatenate((y_p, y_n), 0)
    permutation = np.random.permutation(train_x.shape[1])
    train_x = train_x[:, permutation, :]
    train_y = train_y[permutation]

    train_x = train_x[list(np.where(np.asarray(channel_map) == 1)[0]), :, :]

    k_folds = 10
    model = train_pca_rda_kde_model(train_x, train_y, k_folds=k_folds)

    # Define task and operate
    task_list = [('I_LOVE_COOKIES', 'I_LOVE_'),
                 ('THIS_IS_A_DEMO', 'THIS_IS_A_')]

    task = CopyPhraseWrapper(min_num_seq=1, max_num_seq=25, signal_model=model,
                             fs=dim_x * 2, k=1, alp=alphabet(),
                             task_list=task_list)

    print(task)

if __name__ == '__main__':
    demo_copy_phrase_wrapper()
