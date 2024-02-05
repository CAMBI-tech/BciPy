import logging
import pickle
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from bcipy.helpers.language_model import init_language_model, histogram, with_min_prob
from bcipy.helpers.symbols import alphabet, BACKSPACE_CHAR
from bcipy.language import LanguageModel
from bcipy.signal.model import SignalModel
from bcipy.simulator.helpers.state_manager import SimState
from bcipy.simulator.helpers.types import SimEvidence
from bcipy.task.data import EvidenceType

log = logging.getLogger(__name__)


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

        return dict(sm=SimEvidence(EvidenceType.ERP, eeg_evidence, stimuli))

    def get_model(self, key=None):
        return self.signal_model


class SigLmModelHandler1(ModelHandler):

    def __init__(self, sm_model_file, parameters):
        """

        Args:
            sm_model_file: path to signal model pickle
            parameters: only needs to contain which type of language model
        """
        self.sm_model_file = sm_model_file
        self.signal_model: Optional[SignalModel] = None
        self.lm_model: LanguageModel = init_language_model(parameters)
        self.alp = alphabet()
        self.backspace_prob: float = 0.05

        with open(self.sm_model_file, "rb") as signal_file:
            self.signal_model = pickle.load(signal_file)

    def generate_evidence(self, state: SimState, eeg_responses):
        stimuli = state.display_alphabet
        eeg_evidence = self.signal_model.predict(eeg_responses, stimuli, self.alp)
        lm_evidence = self.lm_model.predict(
            list(state.current_sentence)) if state.inquiry_n == 0 else None

        ret_evidence = {}
        if lm_evidence is not None:

            # adjusting evidence to include backspace prob
            if BACKSPACE_CHAR in stimuli:
                lm_evidence = with_min_prob(lm_evidence, (BACKSPACE_CHAR, self.backspace_prob))

            ret_evidence['lm'] = SimEvidence(EvidenceType.LM, np.array(lm_evidence), self.alp)
            log.debug(f"LM evidence for '{state.current_sentence}' -> \n {histogram(lm_evidence)}")

        if eeg_evidence is not None:
            ret_evidence['sm'] = SimEvidence(EvidenceType.ERP, eeg_evidence, stimuli)

        return ret_evidence

    def get_model(self, key=None):
        if key == 'sm':
            return self.signal_model
        elif key == 'lm':
            return self.lm_model
        else:
            raise RuntimeError(f"Can't find model {key}")
