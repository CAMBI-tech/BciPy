from collections import Counter
import torch
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
import heapq

from bcipy.helpers.symbols import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType

from bcipy.helpers.exceptions import InvalidLanguageModelException

from scipy.special import logsumexp
from scipy.special import softmax

from bcipy.config import LM_PATH

from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer, ByGPT5Config
from transformers.pipelines.text_generation import TextGenerationPipeline

class CausalByteLanguageModel(LanguageModel):
    """Character byte-level language model based on a pre-trained causal model, e.g. ByGPT5"""

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
                 ):
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
        self.vocab = {}
        self.index_to_word = {}
        self.index_to_word_lower = {}
        self.result_to_vocab_indexes = []
        self.symbol_set_lower = None
        self.device = lm_device
        self.left_context = lm_left_context
        self.fp16 = fp16
        self.mixed_case_context = mixed_case_context
        self.case_simple = case_simple

        # Taken from: https://github.com/potamides/uniformers/blob/main/examples/inference/lm_perplexity.py
        # We need to add this to be able to use ByGPT5 with AutoModel
        CONFIG_MAPPING.register(ByGPT5Config.model_type, ByGPT5Config)
        MODEL_FOR_CAUSAL_LM_MAPPING.register(ByGPT5Config, ByGPT5LMHeadModel)
        TOKENIZER_MAPPING.register(ByGPT5Config, (ByGPT5Tokenizer, None))

        # And this too, if we want to test the raw ByT5 decoder
        MODEL_FOR_CAUSAL_LM_MAPPING.register(T5Config, ByGPT5LMHeadModel)

        # We optionally load the model from a local directory, but if this is not
        # specified, we load a Hugging Face model

        causal_params = self.parameters['causal']
        self.model_name = lang_model_name or causal_params['model_name']['value']

        local_model_path = lm_path or causal_params['model_path']['value']
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

        # List of empty lists
        self.result_to_vocab_indexes = [ [] for _ in range(len(self.symbol_set_lower)) ]

        #print(f"DEBUG symbol_set_lower {self.symbol_set_lower}")

        for i in range(self.vocab_size):
            word = self.tokenizer.decode([i])
            #print(f"DEBUG vocab {i} '{word}'")
            word_lower = word.lower()
            self.index_to_word[i] = word
            self.index_to_word_lower[i] = word_lower
            # Create a mapping between the vocab index and the index in the result set
            try:
                # Special case for space
                if word == " ":
                    self.result_to_vocab_indexes[self.symbol_set_lower.index(SPACE_CHAR)].append(i)
                elif word != SPACE_CHAR:
                    self.result_to_vocab_indexes[self.symbol_set_lower.index(word_lower)].append(i)
            except ValueError:
                pass

        #print(f'DEBUG map {self.result_to_vocab_indexes}')

        # Get the index we use for the start or end pseudo-word
        if self.left_context == "":
            self.left_context = "</s>"

        # Get token id(s) for the left context we condition all sentences on
        self.left_context_tokens = self._encode(self.left_context)
        #print(f"DEBUG left_context_tokens = {self.left_context_tokens}")

    def _encode(self, text: str) -> List[int]:
        #print(f"DEBUG _encode '{text}'")
        tokens = self.tokenizer.encode(text)
        if len(tokens) > 1 and self.model_name.startswith("facebook/opt"):
            # Some models always add </s> at start which we don't want since we may have our own left context
            tokens = tokens[1:]
        elif len(tokens) > 1 and self.model_name.startswith("google/byt5"):
            # Some models always add </s> at end
            #print(f"DEBUG '{text}' shorter1 {tokens}")
            tokens = tokens[:-1]
            #print(f"DEBUG '{text}' shorter2 {tokens}")

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

        context = "".join(evidence).replace(SPACE_CHAR, ' ')

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

        # Lower case context if we aren't doing mixed case conditioning
        if not self.mixed_case_context:
            context = context.lower()

        tokens = []
        tokens.extend(self.left_context_tokens)
        # Don't extend if the context is empty, this avoids some models like byt5 from adding extra </s> at start
        if len(context) > 0:
            tokens.extend(self._encode(context))

        #print(f"DEBUG tokens {tokens}")

        tensor = torch.tensor([tokens]).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).logits # Shape is (1, 1, 384)
            log_probs = torch.log_softmax(logits[-1, -1, :], dim=0).to("cpu")

       #print(f"DEBUG: logits {logits} {logits.shape}")
       # print(f"DEBUG: log_probs {log_probs} {log_probs.shape}")

        # Create a simple list with the probabilities of all the characters we need to return
        char_probs = []
        for i in range(len(self.symbol_set_lower)):
            # List of 1 or more indexes in the LLM vocab we need to sum
            indexes = self.result_to_vocab_indexes[i]
            if len(indexes) == 1:
                char_probs.append(float(log_probs[indexes[0]]))
            elif len(indexes) > 1:
                # Create a list of the log probs for this character
                char_log_probs = []
                for index in indexes:
                    char_log_probs.append(log_probs[index])
                char_probs.append(logsumexp(char_log_probs))
            else:
                # This should only happen if the language model doesn't have all our characters
                char_probs.append(float("-inf"))

        # Normalize to a distribution that sums to 1
        char_probs = softmax(char_probs)
        #print(f"DEBUG after {char_probs} {sum(char_probs)}")

        # Now construct the return dictionary that maps the character to its probability
        next_char_pred = Counter()
        for i, ch in enumerate(self.symbol_set_lower):
            if ch is SPACE_CHAR:
                next_char_pred[ch] = char_probs[i]
            else:
                next_char_pred[ch.upper()] = char_probs[i]
        next_char_pred[BACKSPACE_CHAR] = 0.0

        return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))

        #char_probs = softmax(char_probs)
        #next_char_pred = Counter()
        #for i, ch in enumerate(self.symbol_set):
        #    next_char_pred[ch] = char_probs[i]
        #return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))

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
            #self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
            self.model = ByGPT5LMHeadModel.from_pretrained(self.model_dir)
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
