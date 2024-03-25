import logging
import pickle
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from bcipy.helpers.language_model import (histogram, init_language_model,
                                          with_min_prob, language_models_by_name)
from bcipy.helpers.symbols import BACKSPACE_CHAR, alphabet
from bcipy.language import LanguageModel, ResponseType
from bcipy.signal.model import SignalModel
from bcipy.simulator.helpers.state_manager import SimState
from bcipy.simulator.helpers.types import SimEvidence
from bcipy.task.data import EvidenceType
import inspect

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


def sim_init_lm(parameters) -> LanguageModel:
    """
    Init Language Model configured in the sim parameters.
    Mostly copied from language_model.init_lang_model

    Parameters
    ----------
        parameters : dict
            configuration details and path locations

    Returns
    -------
        instance of a LanguageModel
    """

    lm_classes = language_models_by_name()
    model_class: LanguageModel.__class__ = lm_classes[parameters.get("sim_lm_type", "UNIFORM")]
    # introspect the model arguments to determine what parameters to pass.
    args = inspect.signature(model_class).parameters.keys()

    # select the relevant parameters into a dict.
    params = {key: parameters[key] for key in args & parameters.keys()}
    return model_class(response_type=ResponseType.SYMBOL,
                       symbol_set=alphabet(parameters),
                       **params)


class SigLmModelHandler1(ModelHandler):

    def __init__(self, sm_model_file, parameters):
        """

        Args:
            sm_model_file: path to signal model pickle
            parameters: only needs to contain which type of language model
        """
        self.sm_model_file = sm_model_file
        self.signal_model: Optional[SignalModel] = None
        self.lm_model: LanguageModel = sim_init_lm(parameters)
        log.info(f"Language Model Type: {self.lm_model.name()}")
        self.alp = alphabet()
        self.backspace_prob: float = parameters.get('lm_backspace_prob', 0.05)

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
