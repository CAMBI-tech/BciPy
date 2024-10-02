from collections import Counter
from typing import Optional, List, Tuple
from bcipy.helpers.symbols import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.exceptions import InvalidLanguageModelException, KenLMInstallationException
from bcipy.config import LM_PATH
try:
    import kenlm
except BaseException:
    raise KenLMInstallationException(
        "Please install the requisite kenlm package:\n'pip install kenlm==0.1 --global-option=\"--max_order=12\"")
import numpy as np


class KenLMLanguageModel(LanguageModel):
    """Character n-gram language model using the KenLM library for querying"""

    def __init__(self, response_type: ResponseType, symbol_set: List[str], lm_path: Optional[str] = None):

        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.model = None
        kenlm_params = self.parameters['kenlm']
        kenlm_model = kenlm_params['model_file']['value']
        self.lm_path = lm_path or f"{LM_PATH}/{kenlm_model}"

        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    def predict(self, evidence: List[str]) -> List[Tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of
        the next symbol
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbols with probability
        """

        # Do not modify the original parameter, could affect mixture model
        context = evidence.copy()

        if len(context) > 11:
            context = context[-11:]

        evidence_str = ''.join(context).lower()

        for i, ch in enumerate(context):
            if ch == SPACE_CHAR:
                context[i] = "<sp>"

        self.model.BeginSentenceWrite(self.state)

        # Update the state one token at a time based on evidence, alternate states
        for i, token in enumerate(context):
            if i % 2 == 0:
                self.model.BaseScore(self.state, token.lower(), self.state2)
            else:
                self.model.BaseScore(self.state2, token.lower(), self.state)

        next_char_pred = None

        # Generate the probability distribution based on the final state
        if len(context) % 2 == 0:
            next_char_pred = self.prob_dist(self.state)
        else:
            next_char_pred = self.prob_dist(self.state2)

        return next_char_pred

    def update(self) -> None:
        """Update the model state"""
        ...

    def load(self) -> None:
        """
            Load the language model, initialize state variables
        Args:
            path: language model file path
        """

        try:
            self.model = kenlm.LanguageModel(self.lm_path)
        except BaseException:
            raise InvalidLanguageModelException(
                f"A valid model path must be provided for the KenLMLanguageModel.\nPath{self.lm_path} is not valid.")

        self.state = kenlm.State()
        self.state2 = kenlm.State()

    def state_update(self, evidence: List[str]) -> List[Tuple]:
        """
            Wrapper method that takes in evidence text and outputs probability distribution
            of next character
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbols with probabilities
        """
        next_char_pred = self.predict(evidence)

        return next_char_pred

    def prob_dist(self, state: kenlm.State) -> List[Tuple]:
        """
            Take in a state and generate the probability distribution of next character
        Args:
            state - the kenlm state updated with the evidence
        Response:
            A list of symbols with probability
        """
        next_char_pred = Counter()

        temp_state = kenlm.State()

        for char in self.symbol_set:
            # Backspace probability under the LM is 0
            if char == BACKSPACE_CHAR:
                next

            score = 0.0

            # Replace the space character with KenLM's <sp> token
            if char == SPACE_CHAR:
                score = self.model.BaseScore(state, '<sp>', temp_state)
            else:
                score = self.model.BaseScore(state, char.lower(), temp_state)

            # BaseScore returns log probs, convert by putting 10 to its power
            next_char_pred[char] = pow(10, score)

        sum = np.sum(list(next_char_pred.values()))
        for char in self.symbol_set:
            next_char_pred[char] /= sum

        return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))
