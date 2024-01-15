from typing import List, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from bcipy.helpers.symbols import BACKSPACE_CHAR, SPACE_CHAR, alphabet
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.helpers.exceptions import InvalidLanguageModelException
import torch
from scipy.special import logsumexp
from scipy.special import softmax
from collections import Counter


class Seq2SeqLanguageModel(LanguageModel):
    """Transformer seq2seq models like the ByT5 byte-level pretrained model by Google."""

    def __init__(self,
                 response_type: ResponseType,
                 symbol_set: List[str],
                 lang_model_name: str,
                 lm_path: str = None,
                 lm_device: str = "cpu",
                 lm_left_context: str = "",
                 ):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type    - SYMBOL only
            symbol_set       - list of symbol strings
            lang_model_name  - name of the Hugging Face casual language model to load
            lm_path          - load fine-tuned model from specified directory
            lm_device        - device to use for making predictions (cpu, mps, or cuda)
            lm_left_context  - text to condition start of sentence on
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
        self.model_name = lang_model_name
        self.symbol_set_lower = None
        self.symbol_set_lower_ascii = None
        self.device = lm_device
        self.left_context = lm_left_context
        self.left_context_tokens = []

        # Not sure if this matters for performance or accuracy?
        self.num_results = 64

        # We optionally load the model from a local directory, but if this is not specified, we load a Hugging Face model
        self.model_dir = lm_path or lang_model_name

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

        assert self.model is not None, "language model does not exist!"

        context = "".join(evidence).replace(SPACE_CHAR, ' ').lower()

        input_ids = self.tokenizer(context).input_ids

        # After span token, add a space and </s>
        #input_ids_with_span = torch.tensor([input_ids[0:-1] + [258] + [35] + [1]])
        #print(f"input_ids_with_span, size {len(input_ids_with_span)}, {input_ids_with_span}")

        # Add span token and </s>
        input_ids_with_span = torch.tensor([self.left_context_tokens + input_ids[0:-1] + [258] + [1]])
        print(f"input_ids_with_span, size {len(input_ids_with_span)}, {input_ids_with_span}")

        outputs = self.model.generate(
            input_ids_with_span,
            max_length=3,
            num_beams=self.num_results,
            num_return_sequences=self.num_results,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        # Create a hash mapping each valid following character to a list of log probabilities
        char_to_log_probs = {}
        for i in range(len(outputs.sequences)):
            ascii = int(outputs.sequences[i][2]) - 3
            # Convert to lower
            if 65 <= ascii <= 90:
                ascii += 32
            # Skip symbols we don't care about
            if ascii in self.symbol_set_lower_ascii:
                ch = chr(ascii)

                # Add to the previous log prob for this letter (if any)
                if ch in char_to_log_probs:
                    char_to_log_probs[ch] = logsumexp([char_to_log_probs[ch], float(outputs.sequences_scores[i])])
                else:
                    char_to_log_probs[ch] = float(outputs.sequences_scores[i])

        print(f"DEBUG char_to_log_probs {char_to_log_probs}")

        # Parallel array to symbol_set for storing the marginals
        char_probs = []
        for ch in self.symbol_set_lower:
            # Convert space to the underscore used in BciPy
            if ch == SPACE_CHAR:
                target_ch = ' '
            else:
                target_ch = ch

            # Handle cases when symbols are never seen
            if target_ch in char_to_log_probs:
                char_probs += char_to_log_probs[target_ch],
            else:
                char_probs += 0.0,
        print(f"DEBUG char_probs 1 {char_probs}")

        # Normalize to a distribution that sums to 1
        char_probs = softmax(char_probs)

        print(f"DEBUG char_probs 2 {char_probs}, sum={sum(char_probs)}, len={len(char_probs)}")

        # width = 120
        # print("             " + "_" * width)
        # for i in range(len(self.symbol_set_lower)):
        #     print(f"{self.symbol_set_lower[i]} = {char_probs[i]:4.2e} " + "*" * int(char_probs[i] * width))
        # print("             " + "_" * width)

        next_char_pred = Counter()

        for i, ch in enumerate(self.symbol_set_lower):
            if ch is SPACE_CHAR:
                next_char_pred[ch] = char_probs[i]
            else:
                next_char_pred[ch.upper()] = char_probs[i]

        next_char_pred[BACKSPACE_CHAR] = 0.0

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
        except:
            raise InvalidLanguageModelException(f"{self.model_name} is not a valid model identifier on HuggingFace.")
        self.vocab_size = self.tokenizer.vocab_size
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        except:
            raise InvalidLanguageModelException(f"{self.model_dir} is not a valid local folder or model identifier on HuggingFace.")

        self.model.eval()
        self.model.to(self.device)

        self.symbol_set_lower = []
        self.symbol_set_lower_ascii = []
        for ch in self.symbol_set:
            if ch is SPACE_CHAR:
                self.symbol_set_lower.append(SPACE_CHAR)
                self.symbol_set_lower_ascii.append(32)
            elif ch is BACKSPACE_CHAR:
                continue
            else:
                self.symbol_set_lower.append(ch.lower())
                self.symbol_set_lower_ascii.append(ord(ch.lower()))

        # Get token id(s) for the left context we condition all sentences on
        if self.left_context and len(self.left_context) > 0:
            # Tokenizer adds </s> to the end but nothing to the front, drop the end </s>
            self.left_context_tokens = self.tokenizer(self.left_context).input_ids[0:-1]
        print(f"DEBUG left_context_tokens = {self.left_context_tokens}")

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
