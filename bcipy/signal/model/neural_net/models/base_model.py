from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor
from torch.nn import Module


class Classifier(Module, ABC):
    def __init__(self, *args, **kwargs):
        """
        Required attributes:
        - self.device - model must send itself to device at the end of __init__
        - self.criterion
        """
        super().__init__()

    @abstractmethod
    def forward(self, data) -> Tensor:
        """Compute log probabilities of each class for batch"""
        ...

    @abstractmethod
    def _trace(self, example_input):
        """Use torch.jit.trace to replace the actual nn.Module with a ScriptModule"""
        ...

    @abstractmethod
    def get_outputs(self, data, labels) -> Dict[str, Tensor]:
        """
        Send items to self.device
        Run forward pass
        Return dictionary of outputs, e.g. {"loss": loss, "log_probs": log_probs}
        """
        ...

    @abstractmethod
    def get_acc(self, log_probs, labels) -> Tensor:
        """Returns the top-1 accuracy as percentage (values in [0, 100])"""
        ...
