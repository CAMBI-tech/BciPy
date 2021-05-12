from .base_model import Classifier
from .dummy_classifier import DummyEEGClassifier, RiggedClassifier
from .resnet import ResNet1D

__all__ = [
    "Classifier",
    "DummyEEGClassifier",
    "ResNet1D",
    "RiggedClassifier",
]
