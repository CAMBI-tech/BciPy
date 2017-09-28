""" Demo that shows the model overfits to the given training data.
 Observe that the training procedure for the pipeline should be changed to
 make it a proper training. Right now it uses all the training data and does
 not cross validate. """

import numpy as np
import pandas
import sys
from sklearn.model_selection import train_test_split
from sklearn import metrics

sys.path.append('C:\Users\Aziz\Desktop\GIT\TMbci\utils')
sys.path.append('C:\Users\Aziz\Desktop\GIT\TMbci\eeg_model\mach_learning_dir')
sys.path.append('C:\Users\Aziz\Desktop\GIT\TMbci\eeg_model\mach_learning_dir'
                '\classifier')
sys.path.append('C:\Users\Aziz\Desktop\GIT\TMbci\eeg_model\mach_learning_dir'
                '\dimensionality_reduction')
from triggerdecoder import trigger_decoder
from wrapper import PipelineWrapper
from function_classifier import RegularizedDiscriminantAnalysis
from function_dim_reduction import PCA

folder_name = 'C:\Users\Aziz\Desktop\Junk/tab_sample.csv'
info_frame = pandas.read_csv(folder_name, engine='python', nrows=1)
dat_frame = pandas.read_csv(folder_name, engine='python', skiprows=[0, 1])
dat_o = dat_frame.values

fs = int(info_frame.values[0][1])
size_test = 0.1

trig_part = {'windowLengthinSamples': fs / 2, 'sequenceEndID': 31,
             'pauseID': 35, 'fixationID': 29, 'TARGET_TRIGGER_OFFSET': 128}

len_trial = trig_part['windowLengthinSamples']
down_rate = 2
dat = dat_o[0::down_rate, :]

eeg_dat = dat_o[:, 0:20]
trigger = dat_o[:, -1]
processed_trig = trigger_decoder(trigger, trig_part)

tar_trial = processed_trig[3]
idx_time_trial = processed_trig[1]
num_seq = idx_time_trial.shape[0]
num_trial = idx_time_trial.shape[1]

dat_trial = []
for idx_seq in range(idx_time_trial.shape[0]):
    for idx_tri in range(idx_time_trial.shape[1]):
        t_start = idx_time_trial[idx_seq, idx_tri]
        dat_trial.append(eeg_dat[t_start:t_start + len_trial, :])
dat_trial = np.array(dat_trial)

feature_trial = np.array(
    [np.concatenate(dat_trial[i, :, :]) for i in range(dat_trial.shape[0])])
tar_feature_trial = np.concatenate(tar_trial)

pipeline = PipelineWrapper()
rda = RegularizedDiscriminantAnalysis()
pca = PCA()
pipeline.add(pca)
pipeline.add(rda)

# train_x, test_x, train_y, test_y = train_test_split(
#     feature_trial, tar_feature_trial, test_size=size_test)

len_x = feature_trial.shape[0]
train_x = feature_trial[0:int(len_x * (1 - size_test)), :]
train_y = tar_feature_trial[0:int(len_x * (1 - size_test))]
test_x = feature_trial[int(len_x * (1 - size_test)):, :]
test_y = tar_feature_trial[int(len_x * (1 - size_test)):len_x]

sc_t = pipeline.fit_transform(train_x, train_y)
sc_t = np.dot(np.array([-1, 1]), sc_t.transpose())
sc = pipeline.transform(test_x)
sc = np.dot(np.array([-1, 1]), sc.transpose())

fpr, tpr, _ = metrics.roc_curve(train_y, sc_t, pos_label=1)
train_auc = metrics.auc(fpr, tpr)
fpr, tpr, _ = metrics.roc_curve(test_y, sc, pos_label=1)
test_auc = metrics.auc(fpr, tpr)
print("Train AUC: {}, Test AUC: {}".format(train_auc, test_auc))
print("Parameters lam/gam: {}/{}".format(pipeline.pipeline[1].lam,
      pipeline.pipeline[1].gam))
