import pickle
from abc import ABC, abstractmethod
from typing import Optional

from bcipy.helpers.symbols import alphabet
from bcipy.signal.model import SignalModel
from bcipy.simulator.helpers.state_manager import SimState


class ModelHandler(ABC):

    @abstractmethod
    def generate_evidence(self, state: SimState, features):
        """ Run model on features to generate evidence """

    @abstractmethod
    def get_model(self, key=None):
        """ get the model """


class SignalModelHandler1(ModelHandler):

    def __init__(self, model_file):
        self.model_file = model_file
        self.signal_model: Optional[SignalModel] = None
        with open(self.model_file, "rb") as signal_file:
            self.signal_model = pickle.load(signal_file)

    def generate_evidence(self, state: SimState, features):
        stimuli = state.display_alphabet
        alp = alphabet()
        eeg_evidence = self.signal_model.predict(features, stimuli, alp)

        return eeg_evidence

    def get_model(self, key=None):
        return self.signal_model
