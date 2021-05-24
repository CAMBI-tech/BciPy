import torch
import torch.nn.functional as F

from bcipy.signal.model.neural_net.models.base_model import Classifier


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
