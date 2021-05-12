import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ELU, BatchNorm1d, Conv1d, Flatten, Linear, LogSoftmax, Sequential

from .base_model import Classifier


class DummyEEGClassifier(Classifier):
    """Useful for smoke testing only"""

    def __init__(self, in_chan, duration, n_classes, device, criterion=F.nll_loss):
        """
        :param in_chan: number of channels in the raw data
        :param duration: number of time points in the raw data
        :param n_classes: perform N-way classification
        """
        super().__init__()
        self.in_chan = in_chan
        self.duration = duration
        self.n_classes = n_classes
        self.device = device

        self.criterion = criterion

        # TODO - use a real network structure
        # in EEGNet, they use "DepthwiseConv2D" from Keras -
        # it looks like what we actually want is just a grouped conv1d, but then
        # their choice of layer seems weird. Need to understand better...
        self.conv = Sequential(
            Conv1d(in_chan, 32, kernel_size=5, stride=2, padding=0, bias=False),
            BatchNorm1d(32),
            ELU(),
        )

        dummy_input = torch.zeros(1, in_chan, duration)
        shape_after_conv = np.prod(self.conv(dummy_input).shape[1:])

        self.classifier = Sequential(
            Flatten(),
            Linear(shape_after_conv, n_classes),
            LogSoftmax(dim=-1),
        )
        self.to(self.device)

    def forward(self, data):
        features = self.conv(data)
        log_probs = self.classifier(features)
        return log_probs

    def get_outputs(self, data, labels):
        data = data.to(self.device)
        labels = labels.to(self.device)

        log_probs = self.forward(data)
        loss = self.criterion(log_probs, labels)
        acc = self.get_acc(log_probs, labels)
        return {"loss": loss, "log_probs": log_probs, "acc": acc}

    def get_acc(self, log_probs, labels):
        return torch.tensor(100 * log_probs.argmax(1).eq(labels).sum().item() / labels.shape[0])

    def _trace(self):
        raise NotImplementedError()


class RiggedClassifier(Classifier):
    """Ignores data and gives deterministic results."""

    def __init__(
        self,
        indices: torch.Tensor,
        probs: torch.Tensor,
        other_seen_probs: torch.Tensor,
        unseen_class_probs: torch.Tensor,
        seq_len: int,
        none_class_idx: int,
        criterion=F.nll_loss,
    ):
        """
        put `probs` mass onto `indices`
        put `other_seen_probs` mass onto all other seen letters
        put `unseen_class_probs` onto the "none" class located at `none_class_idx`
        """
        super().__init__()
        self.indices = indices
        self.probs = probs
        self.other_seen_probs = other_seen_probs
        self.unseen_class_probs = unseen_class_probs
        self.seq_len = seq_len
        self.none_class_idx = none_class_idx
        self.criterion = criterion

    def forward(self, data):
        batch_size = data.shape[0]
        return (
            self.other_seen_probs.expand(batch_size, self.seq_len + 1)
            .scatter(1, self.indices, self.probs)
            .scatter(1, torch.tensor([self.none_class_idx]).repeat(batch_size, 1), self.unseen_class_probs)
            .log()
        )

    def get_outputs(self, data, labels):
        log_probs = self(data)
        loss = self.criterion(log_probs, labels)
        acc = self.get_acc(log_probs, labels)
        return {"loss": loss, "log_probs": log_probs, "acc": acc}

    def get_acc(self, log_probs, labels):
        return torch.tensor(100 * log_probs.argmax(1).eq(labels).sum().item() / labels.shape[0])

    def _trace(self):
        raise NotImplementedError()
