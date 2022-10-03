import random
import shutil
import tempfile
import unittest

from pathlib import Path

import numpy as np

from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.acquisition.devices import DeviceSpec, register
from bcipy.helpers.copy_phrase_wrapper import CopyPhraseWrapper
from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.task import alphabet
from bcipy.language.uniform import UniformLanguageModel
from bcipy.signal.model import PcaRdaKdeModel
from bcipy.task.data import EvidenceType


class TestCopyPhraseWrapper(unittest.TestCase):
    """Test CopyPhraseWrapper"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        random.seed(0)
        cls.tmp_dir = Path(tempfile.mkdtemp())

        cls.device_spec = DeviceSpec(name="dummy_device", channels=["1", "2", "3"], sample_rate=10)
        register(cls.device_spec)

        cls.params_used = DEFAULT_PARAMETERS_PATH
        cls.params = load_json_parameters(cls.params_used, value_cast=True)

        # # Generate fake data and train a model
        # Specify data dimensions
        cls.num_trial = cls.params["stim_length"]
        cls.dim_x = 5
        cls.num_channel = cls.device_spec.channel_count
        cls.num_x_pos = 200
        cls.num_x_neg = 200

        # Generate Gaussian random data
        # TODO - deduplicate, add to test utils somewhere (data generation matching a device_spec)
        cls.pos_mean, cls.pos_std = 0, 0.5
        cls.neg_mean, cls.neg_std = 1, 0.5
        x_pos = cls.pos_mean + cls.pos_std * np.random.randn(cls.num_channel, cls.num_x_pos, cls.dim_x)
        x_neg = cls.neg_mean + cls.neg_std * np.random.randn(cls.num_channel, cls.num_x_neg, cls.dim_x)
        y_pos = np.ones(cls.num_x_pos)
        y_neg = np.zeros(cls.num_x_neg)

        # Stack and permute data
        x = np.concatenate([x_pos, x_neg], 1)
        y = np.concatenate([y_pos, y_neg], 0)
        permutation = np.random.permutation(cls.num_x_pos + cls.num_x_neg)
        x = x[:, permutation, :]
        y = y[permutation]

        cls.model = PcaRdaKdeModel(k_folds=10)
        cls.model.fit(x, y)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_valid_letters(self):
        alp = alphabet()
        cp = CopyPhraseWrapper(
            min_num_inq=1,
            max_num_inq=50,
            lmodel=None,
            signal_model=None,
            fs=25,
            k=2,
            alp=alp,
            task_list=[("HELLO_WORLD", "HE")],
            is_txt_stim=True,
            device_name=self.device_spec.name,
            evidence_names=[EvidenceType.LM, EvidenceType.ERP],
            device_channels=self.device_spec.channels,
            stim_timing=[0.5, 0.25],
        )

        triggers = [
            ("+", 0.0),
            ("H", 0.5670222830376588),
            ("D", 0.8171830819919705),
            ("J", 1.0843321380089037),
            ("B", 1.3329724550130777),
            ("C", 1.5825864360085689),
            ("A", 1.833380013005808),
            ("F", 2.083211077027954),
            ("G", 2.333359022042714),
            ("I", 2.583265081048012),
            ("E", 2.833274284028448),
        ]
        target_info = [
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
        ]

        letters, timings, labels = cp.letter_info(triggers, target_info)
        expected_letters = ["H", "D", "J", "B", "C", "A", "F", "G", "I", "E"]
        expected_time = [
            0.5670222830376588,
            0.8171830819919705,
            1.0843321380089037,
            1.3329724550130777,
            1.5825864360085689,
            1.833380013005808,
            2.083211077027954,
            2.333359022042714,
            2.583265081048012,
            2.833274284028448,
        ]
        self.assertEqual(expected_letters, letters)
        self.assertEqual(expected_time, timings)
        self.assertEqual(len(letters), len(labels))

        triggers = [
            ("+", 0.1),
            ("H", 0.5670222830376588),
            ("D", 0.8171830819919705),
            ("J", 1.0843321380089037),
            ("B", 1.3329724550130777),
            ("C", 1.5825864360085689),
            ("A", 1.833380013005808),
            ("F", 2.083211077027954),
            ("G", 2.333359022042714),
            ("I", 2.583265081048012),
            ("E", 2.833274284028448),
        ]
        target_info = [
            "fixation",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
        ]
        letters, timings, labels = cp.letter_info(triggers, target_info)
        self.assertEqual(expected_letters, letters)
        self.assertEqual(expected_time, timings)
        self.assertEqual(["nontarget"] * (len(letters)), labels)

        # Test it throws an exception when letter is outside alphabet
        with self.assertRaises(Exception):
            cp.letter_info([("A", 0.0), ("*", 1.0)], ["nontarget", "nontarget"])

    def test_init_series_evaluate_inquiry(self):
        alp = alphabet()

        # Create fake data to provide as the user's response
        duration = int(self.num_trial * self.device_spec.sample_rate * self.params["trial_length"])
        response_eeg = self.pos_mean + self.pos_std * np.random.randn(self.num_channel, duration)

        # Note that frequencies for notch and bandpass filter below are chosen to be < 1/2 of sampling frequency
        # due to limitations of nyquist-shannon sampling theorem
        copy_phrase_task = CopyPhraseWrapper(
            min_num_inq=1,
            max_num_inq=50,
            lmodel=UniformLanguageModel(symbol_set=alp),
            signal_model=self.model,
            fs=self.device_spec.sample_rate,
            k=1,
            alp=alp,
            task_list=[("HELLO_WORLD", "HE")],
            is_txt_stim=True,
            device_name=self.device_spec.name,
            evidence_names=[EvidenceType.LM, EvidenceType.ERP],
            device_channels=self.device_spec.channels,
            stim_timing=[0.5, 0.25],
            notch_filter_frequency=4.0,
            filter_low=1.0,
            filter_high=4.0,
            stim_length=self.params["stim_length"],
        )

        is_accepted, sti = copy_phrase_task.initialize_series()
        self.assertFalse(is_accepted)

        triggers = [
            ("+", 0.0),
            ("H", 0.5670222830376588),
            ("D", 0.8171830819919705),
            ("J", 1.0843321380089037),
            ("B", 1.3329724550130777),
            ("C", 1.5825864360085689),
            ("A", 1.833380013005808),
            ("F", 2.083211077027954),
            ("G", 2.333359022042714),
            ("I", 2.583265081048012),
            ("E", 2.833274284028448),
        ]
        stimuli = [label for (label, _) in triggers]
        target_info = [
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
        ]

        new_series, inquiry = copy_phrase_task.evaluate_inquiry(
            response_eeg, triggers, target_info, self.params["trial_length"]
        )
        self.assertFalse(new_series)
        self.assertEqual(sorted(inquiry.stimuli[0]), sorted(stimuli))
        self.assertEqual(inquiry.durations[0], [self.params["time_fixation"]] +
                         [self.params["time_flash"]] * self.params["stim_length"])
        self.assertEqual(inquiry.colors[0], [self.params["fixation_color"]] +
                         [self.params["stim_color"]] * self.params["stim_length"])


if __name__ == '__main__':
    unittest.main()
