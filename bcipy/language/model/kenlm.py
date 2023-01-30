from collections import Counter
from typing import Dict, List, Tuple
from bcipy.language.main import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType
import kenlm

class KenLMLanguageModel(LanguageModel):
    """12-gram character language model based on KenLM"""

    def __init__(self, response_type: ResponseType, symbol_set: List[str], lm_path: str = None):
        
        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.model = None
        self.lm_path = lm_path or "../lms/lm_dec19_char_12gram_1e-5_kenlm_probing.bin"

        self.cache = {}

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

        if len(evidence) > 11:
            evidence = evidence[-11:]

        evidence_str = ''.join(evidence)

        for i, ch in enumerate(evidence):
            if ch == SPACE_CHAR:
                evidence[i] = "<sp>"

        # cache_state = self.check_cache(evidence_str)

        # if cache_state is None:

        self.model.BeginSentenceWrite(self.state)
        
        # Update the state one token at a time based on evidence, alternate states
        for i, token in enumerate(evidence):
            if i % 2 == 0:
                self.model.BaseScore(self.state, token, self.state2)
            else:
                self.model.BaseScore(self.state2, token, self.state)

        next_char_pred = None

        # Generate the probability distribution based on the final state, save state to cache
        if len(evidence) % 2 == 0:
            next_char_pred = self.prob_dist(self.state)
            self.cache[evidence_str] = self.state
        else:
            next_char_pred = self.prob_dist(self.state2)
            self.cache[evidence_str] = self.state2

        # else:
            # next_char_pred = self.prob_dist(cache_state)

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
        
        self.model = kenlm.LanguageModel(self.lm_path)

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

        return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))

    def clear_cache(self):
        """Clear the internal cache of states"""

        self.cache = {}

    def check_cache(self, evidence_str: str) -> kenlm.State:
        if evidence_str in self.cache.keys():
            return self.cache[evidence_str]

        if evidence_str[:-1] in self.cache.keys():
            temp_state = self.cache[evidence_str[:-1]]
            new_state = kenlm.State()
            self.model.BaseScore(temp_state, evidence_str[-1], new_state)
            self.cache[evidence_str] = new_state
            return new_state

        arr = [evidence_str[:-1] == x[1:] for x in self.cache.keys()]
        if any(arr) and len(evidence_str) >= 11:

            temp_state = list(self.cache.values())[arr.index(True)]
            new_state = kenlm.State()
            self.model.BaseScore(temp_state, evidence_str[-1], new_state)
            self.cache[evidence_str] = new_state
            return new_state
        
        return None
            