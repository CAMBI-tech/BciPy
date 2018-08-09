from bcipy.helpers.load import read_data_csv, load_experimental_data, \
    load_json_parameters
from bcipy.signal_processing.sig_pro import sig_pro
from bcipy.signal_model.offline_analysis import offline_analysis
from bcipy.helpers.bci_task_related import trial_reshaper
from bcipy.helpers.triggers import trigger_decoder
from bcipy.helpers.acquisition_related import analysis_channels
import sklearn as sc
import argparse
from bcipy.helpers.load import load_classifier
import os.path

list_model_folder = ['D:/BCIpy/Model-20180807T144701Z-001/Model/',
                     'D:/BCIpy/test_BE_6.14_DSI.LSL_Thu_14_Jun_2018_14hr54min37sec_-0700/']
list_data_folder = ['D:/BCIpy/test_BE_6.14_DSI.LSL_Thu_14_Jun_2018_14hr54min37sec_-0700/',
                    'D:/BCIpy/AbbyTest/AbbyTest_Thu_21_Jun_2018_14hr37min10sec_-0700/',
                    'D:/BCIpy/AbbyTest_Thu_21_Jun_2018_14hr37min10sec_-0700-20180807T154912Z-001\AbbyTest_Thu_21_Jun_2018_14hr37min10sec_-0700/',
                    'D:/BCIpy/Model-20180807T144701Z-001/Model/']

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_folder', default=None)
parser.add_argument('-p', '--parameters_file',
                    default='D:/BCIpy/BciPy/bcipy/parameters/parameters.json')
args = parser.parse_args()

print(f'Loading params from {args.parameters_file}')
parameters = load_json_parameters(args.parameters_file,
                                  value_cast=True)

for model_folder in list_model_folder:
    if os.path.isfile(model_folder + 'model.pkl'):
        model = load_classifier(filename=model_folder + 'model.pkl')
    else:
        model = offline_analysis(model_folder)
    for data_folder in list_data_folder:
        mode = 'calibration'

        raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
            data_folder + '/' + parameters.get('raw_data_name', 'raw_data.csv'))

        # print(f'Channels read from csv: {channels}')
        # print(f'Device type: {type_amp}')

        downsample_rate = parameters.get('down_sampling_rate', 2)
        filtered_data = sig_pro(raw_dat, fs=fs, k=downsample_rate)

        # Process triggers.txt
        triggers_file = parameters.get('triggers_file_name', 'triggers.txt')
        _, t_t_i, t_i, offset = trigger_decoder(mode=mode, trigger_loc=f"{data_folder}/{triggers_file}")

        # Channel map can be checked from raw_data.csv file.
        # read_data_csv already removes the timespamp column.
        channel_map = analysis_channels(channels, type_amp)

        x, y, num_seq, _ = trial_reshaper(t_t_i, t_i, filtered_data,
                                          mode=mode, fs=fs, k=downsample_rate,
                                          offset=offset,
                                          channel_map=channel_map)

        model.transform(x)
        tmp = model.line_el[-2]
        fpr, tpr, thresholds = sc.metrics.roc_curve(y, tmp, pos_label=1)
        auc = sc.metrics.auc(fpr, tpr)
        print("model:{} - data:{} -> AUC:{}".format(model_folder, data_folder, auc))
