"""All models must satisfy the expected model API."""
import unittest

import torch

from bcipy.signal.model.neural_net.data import EEGDataset, data_config, get_fake_data
from bcipy.signal.model.neural_net.models import (
    DummyEEGClassifier,
    ResNet1D,
    RiggedClassifier,
)
from bcipy.signal.model.neural_net.utils import get_default_cfg


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create fake training data
        cls.cfg = get_default_cfg()

        dataset_kw = {
            "N": cls.cfg.batch_size,
            "channels": data_config[cls.cfg.data_device]["n_channels"],  # type: ignore
            "classes": data_config[cls.cfg.data_device][cls.cfg.data_mode]["n_classes"],  # type: ignore
            "length": data_config[cls.cfg.data_device][cls.cfg.data_mode]["length"],  # type: ignore
        }

        cls.train_set = EEGDataset(*get_fake_data(**dataset_kw))

    def check_model_api(self, model, batch_size):
        # Get dummy data item
        data, labels = self.train_set[:batch_size]
        model_outputs = model.get_outputs(data, labels)

        # Check model API:
        # - outputs dict contains correct items
        required_outputs_keys = ["loss", "acc", "log_probs"]
        has_correct_keys = set(required_outputs_keys) == set(model_outputs.keys())

        # - log_probs are normalized
        item_totals = model_outputs["log_probs"].exp().sum(1)
        has_normalized_probs = torch.allclose(item_totals, torch.ones_like(item_totals))

        # - accuracies between [0, 100]
        acc = model_outputs["acc"]
        has_valid_acc_range = torch.all(torch.zeros_like(acc) <= acc) and torch.all(acc <= 100 * torch.ones_like(acc))

        return has_correct_keys and has_normalized_probs and has_valid_acc_range

    def test_ResNet1D(self):
        prod_model = ResNet1D(
            layers=[2, 2, 2, 2],
            num_classes=data_config[self.cfg.data_device][self.cfg.data_mode]["n_classes"],
            in_channels=data_config[self.cfg.data_device]["n_channels"],
            act_name=self.cfg.activation,
            device=self.cfg.device,
        )
        self.assertTrue(self.check_model_api(prod_model, self.cfg.batch_size))

    def test_DummyEEGClassifier(self):
        prod_model = DummyEEGClassifier(
            in_chan=data_config[self.cfg.data_device]["n_channels"],
            duration=data_config[self.cfg.data_device][self.cfg.data_mode]["length"],
            n_classes=data_config[self.cfg.data_device][self.cfg.data_mode]["n_classes"],
            device=self.cfg.device,
        )
        self.assertTrue(self.check_model_api(prod_model, self.cfg.batch_size))

    def test_RiggedClassifier(self):
        n_classes = data_config[self.cfg.data_device][self.cfg.data_mode]["n_classes"]
        seq_len = n_classes - 1
        peak = 0.8
        other_seen = (1 - peak) / (n_classes - 2)
        rest = 1 - peak - (other_seen * (n_classes - 2))

        prod_model = RiggedClassifier(
            indices=torch.tensor([0]).repeat(self.cfg.batch_size, 1),
            probs=torch.tensor([peak]).repeat(self.cfg.batch_size, 1),
            seq_len=seq_len,
            other_seen_probs=torch.tensor([other_seen]).repeat(self.cfg.batch_size, 1),
            unseen_class_probs=torch.tensor([rest]).repeat(self.cfg.batch_size, 1),
            none_class_idx=seq_len,
        )
        self.assertTrue(self.check_model_api(prod_model, self.cfg.batch_size))


if __name__ == "__main__":
    unittest.main()
