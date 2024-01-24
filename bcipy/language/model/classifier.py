from collections import Counter
import torch
from typing import Optional, List, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bcipy.helpers.symbols import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType

from bcipy.helpers.exceptions import InvalidLanguageModelException

from bcipy.config import LM_PATH

from scipy.special import softmax


class ClassifierLanguageModel(LanguageModel):
    """Character language model based on a transformer with a classification head"""

    def __init__(self,
                 response_type: ResponseType,
                 symbol_set: List[str],
                 lang_model_name: Optional[str] = None,
                 lm_path: Optional[str] = None,
                 lm_device: str = "cpu",
                 lm_left_context: str = "",
                 fp16: bool = False,
                 mixed_case_context = False,
                 case_simple = False,
                 **_):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type      - SYMBOL only
            symbol_set         - list of symbol strings
            lang_model_name    - name of the Hugging Face casual language model to load
            lm_path            - load fine-tuned model from specified directory
            lm_device          - device to use for making predictions (cpu, mps, or cuda)
            lm_left_context    - text to condition start of sentence on
            fp16               - convert model to fp16 to save memory/compute on CUDA
            mixed_case_context - use mixed case for language model left context
            case_simple        - simple fixing of left context case
        """
        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.valid_vocab = []
        self.vocab = {}
        self.longest_token = 0
        self.index_to_word = {}
        self.index_to_word_lower = {}
        self.symbol_set_lower = None
        self.device = lm_device
        self.left_context = lm_left_context
        self.fp16 = fp16
        self.mixed_case_context = mixed_case_context
        self.case_simple = case_simple

        # We optionally load the model from a local directory, but if this is not
        # specified, we load a Hugging Face model

        classifier_params = self.parameters['classifier']
        self.model_name = lang_model_name or classifier_params['model_name']['value']

        local_model_path = lm_path or classifier_params['model_path']['value']
        self.model_dir = f"{LM_PATH}/{local_model_path}" if local_model_path != "" else self.model_name

        self.simple_upper_words = {"i": "I",
                                    "i'll": "I'll",
                                    "i've": "I've",
                                    "i'd": "I'd",
                                    "i'm": "I'm"}
        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    def _build_vocab(self) -> None:
        """
        Build a vocabulary table mapping token index to word strings
        """

        for i in range(self.vocab_size):
            word = self.tokenizer.decode([i])
            word_lower = word.lower()
            self.index_to_word[i] = word
            self.index_to_word_lower[i] = word_lower
            valid = True
            for ch in word_lower:
                # The space char is only valid once we convert spaces to the space char
                if ch == SPACE_CHAR:
                    valid = False
                    break
                if ch == ' ':
                    continue
                elif ch not in self.symbol_set_lower:
                    valid = False
                    break
            if valid:
                self.valid_vocab += i,
                length = len(word)
                if length > self.longest_token:
                    self.longest_token = length
                for j in range(length):
                    key = word_lower[0:j + 1].replace(' ', SPACE_CHAR)
                    if key not in self.vocab:
                        self.vocab[key] = []
                    self.vocab[key] += i,

        # Get the index we use for the start or end pseudo-word
        if self.left_context == "":
            if "gpt2" in self.model_name:
                self.left_context = "<|endoftext|>"
            else:
                self.left_context = "</s>"
        # Get token id(s) for the left context we condition all sentences on
        self.left_context_tokens = self._encode(self.left_context)
        # print(f"left_context_tokens = {self.left_context_tokens}")

    def _encode(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        if len(tokens) > 1 and self.model_name.startswith("facebook/opt"):
            tokens = tokens[1:]

        return tokens

    def predict(self, evidence: List[str]) -> List[Tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of
        the next symbol
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbols with probability
        """

        assert self.model is not None, "language model does not exist!"

        converted_context = "".join(evidence)

        context = converted_context.replace(SPACE_CHAR, ' ')

        if self.case_simple and len(context) > 0:
            cased_context = ""
            words = context.split()
            for i, word in enumerate(words):
                if i == 0 and word[0] >= 'a' and word[0] <= 'z':
                    word = word[0].upper() + word[1:]
                if i > 0:
                    if word in self.simple_upper_words:
                        word = self.simple_upper_words[word]
                    cased_context += " "
                cased_context += word
            # Handle ending space in the context
            if context[-1] == ' ':
                cased_context += " "
            #print(f"Simple casing of left context, from '{context}' to '{cased_context}'")
            context = cased_context

        else:
            context = context.lower()

        tokens = []
        tokens.extend(self.left_context_tokens)

        # Optionally, we condition on upper and lower case left context
        if not self.mixed_case_context:
            context = context.lower()
        tokens.extend(self._encode(context))

        tensor = torch.tensor([tokens]).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).logits
            char_probs = torch.softmax(logits, dim=1).to("cpu").numpy()[0]

            # Apostrophe is not in our symbol set, redistribute prob mass
            char_probs[26] = 0.0

        char_probs = softmax(char_probs)

        next_char_pred = Counter()

        for i, ch in enumerate(self.symbol_set):
            next_char_pred[ch] = char_probs[i]

        return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))

    def update(self) -> None:
        """Update the model state"""
        ...

    def load(self) -> None:
        """
            Load the language model and tokenizer, initialize class variables
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        except BaseException:
            raise InvalidLanguageModelException(f"{self.model_name} is not a valid model identifier on HuggingFace.")
        self.vocab_size = self.tokenizer.vocab_size
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            if self.fp16 and self.device == "cuda":
                self.model = self.model.half()
        except:
            raise InvalidLanguageModelException(f"{self.model_dir} is not a valid local folder or model identifier on HuggingFace.")

        self.model.eval()

        self.model.to(self.device)

        self.symbol_set_lower = []
        for ch in self.symbol_set:
            if ch is SPACE_CHAR:
                self.symbol_set_lower.append(SPACE_CHAR)
            elif ch is BACKSPACE_CHAR:
                continue
            else:
                self.symbol_set_lower.append(ch.lower())

        self._build_vocab()

    def state_update(self, evidence: List[str]) -> List[Tuple]:
        """
            Wrapper method that takes in evidence text, and output probability distribution
            of next character
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbol with probability
        """
        next_char_pred = self.predict(evidence)

        return next_char_pred

    def get_num_parameters(self) -> int:
        """
            Find out how many parameters the loaded model has
        Args:
        Response:
            Integer number of parameters in the transformer model
        """
        return sum(p.numel() for p in self.model.parameters())
