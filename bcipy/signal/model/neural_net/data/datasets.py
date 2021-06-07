from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import torch
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import read_data_csv
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.task import InquiryReshaper, Reshaper
from bcipy.helpers.triggers import trigger_decoder
from bcipy.signal.model.base_model import SignalModel
from bcipy.signal.model.neural_net.utils import adjust_length
from bcipy.signal.process import get_default_transform
from loguru import logger
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, transform: Callable = lambda x: x):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.labels[idx]


class FolderDataset(EEGDataset):
    """Load multiple sessions contained in folder. Sessions must use the following layout:

        my_collection/
            session1/
                raw_data.csv
                triggers.txt
                parameters.json
                ...
            session2/
                raw_data.csv
                triggers.txt
                parameters.json
                ...
            ...
    """

    def __init__(self, folder: Union[Path, str], reshaper: Reshaper = InquiryReshaper):
        """
        Loads all data and labels from provided folder.
        TODO - fragile; this code does not enforce consistency of parameters.
            Instead, it just uses parameters from the first folder for all other datasets.

        Args:
            folder (Union[Path, str]): folder containing 1 or more sessions
            reshaper (Reshaper): [description]
        """
        sessions = [x for x in Path(folder).iterdir() if x.is_dir()]
        parameters = Parameters(source=sessions[0] / "parameters.json", cast_values=True)

        trial_length = parameters.get('trial_length')

        downsample_rate = parameters.get('down_sampling_rate', 2)
        notch_filter = parameters.get('notch_filter_frequency', 60)
        hp_filter = parameters.get('filter_high', 45)
        lp_filter = parameters.get('filter_low', 2)
        filter_order = parameters.get('filter_order', 2)

        raw_data_filename = parameters.get('raw_data_name', 'raw_data.csv')
        triggers_filename = parameters.get('trigger_file_name', 'triggers.txt')

        mode = "calibration"

        all_data_list = []
        all_labels_list = []

        for session in sessions:
            raw_data, _, channels, amp_type, fs = read_data_csv(session / raw_data_filename)

            # TODO - annoying to create transform every time, but we need fs
            default_transform = get_default_transform(
                sample_rate_hz=fs,
                notch_freq_hz=notch_filter,
                bandpass_low=lp_filter,
                bandpass_high=hp_filter,
                bandpass_order=filter_order,
                downsample_factor=downsample_rate,
            )
            transformed_data, fs = default_transform(raw_data, fs)

            _, trial_target_info, timing_info, offset = trigger_decoder(
                mode=mode, trigger_path=session / triggers_filename)

            channel_map = analysis_channels(channels, amp_type)

            data, labels = reshaper(
                trial_labels=trial_target_info,
                timing_info=timing_info,
                eeg_data=transformed_data,
                fs=fs,
                trials_per_inquiry=parameters.get('stim_length'),
                offset=offset,
                channel_map=channel_map,
                trial_length=trial_length)

            all_data_list.append(data)
            all_labels_list.append(labels)

        super().__init__(data=torch.stack(all_data_list), labels=torch.stack(all_labels_list))


def get_fake_data(N, channels, classes, length) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(N, channels, length).float()
    y = torch.randint(classes, (N,)).long()
    return x, y


def get_data_from_folder(
    folder: Union[str, Path], length: int, length_tol: int, shuffle=True, quick_test=False, subfolder="sequences"
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_list = []
    labels_list = []

    subjects = [x for x in Path(folder).iterdir() if x.is_dir()]
    for s in subjects:
        try:
            d = torch.load(s / subfolder / "data.pt")
        except FileNotFoundError:
            continue  # TODO - for now, ignore empty folders

        if d.shape[-1] != length:
            if abs(length - d.shape[-1]) <= length_tol:
                d = adjust_length(d, length)
            else:
                continue  # TODO - for now, just skipping other data...

        lab = torch.load(s / subfolder / "labels.pt")
        if lab.shape[0] != d.shape[0]:
            logger.warning(f"SHAPE MISMATCH! subject: {s}, data: {d.shape}, labels: {lab.shape}")
            continue

        data_list.append(d)
        labels_list.append(lab)

        if quick_test and len(data_list) > 10:  # During quick test, break after locating several usable samples
            break

    data = torch.cat(data_list, dim=0).float()
    labels = torch.cat(labels_list, dim=0).long()

    if shuffle:
        r = torch.randperm(data.shape[0])
        data = data[r, ...]
        labels = labels[r, ...]

    return data, labels
