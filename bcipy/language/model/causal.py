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


class CausalLanguageModel(LanguageModel):
    """Character language model based on a pre-trained causal model, GPT-2 by default."""

    def __init__(self,
                 response_type: ResponseType,
                 symbol_set: List[str],
                 lang_model_name: Optional[str] = None,
                 lm_path: Optional[str] = None,
                 lm_device: str = "cpu",
                 lm_left_context: str = "",
                 ):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type - SYMBOL only
            symbol_set    - list of symbol strings
            lang_model_name    - name of the Hugging Face casual language model to load
            lm_path     - load fine-tuned model from specified directory
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
        self.symbol_set_lower = None
        self.device = lm_device
        self.left_context = lm_left_context

        # We optionally load the model from a local directory, but if this is not
        # specified, we load a Hugging Face model

        causal_params = self.parameters['causal']
        self.model_name = lang_model_name or causal_params['model_name']['value']

        local_model_path = lm_path or causal_params['model_path']['value']
        self.model_dir = f"{LM_PATH}/{local_model_path}" if local_model_path != "" else self.model_name

        # parameters for search
        self.beam_width = 8
        self.batch_size = 8

        # backoff to the previous space
        self.token_backoff = -1

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
        converted_context_lower = converted_context.lower()

        context = converted_context.replace(SPACE_CHAR, ' ')
        context_lower = context.lower()

        # Index in the hypothesis string that is the next character after our context
        target_pos = len(context_lower)

        # Remove token_backoff tokens from the end of the context
        # If token_backoff is -1 or goes beyond a word boundary, then
        # search from the last space character in the context
        # If no space, then from the very beginning
        tokens = self._encode(context_lower)

        # Look for the last space in the context, or -1 if no begin_text in context yet
        pos = context_lower.rfind(" ")
        if pos >= 0:
            truncated_context = context_lower[0:pos]
            truncated_tokens = self._encode(truncated_context)
            # Insert the left context tokens at the start of the sequence
            truncated_tokens[0:0] = self.left_context_tokens
        else:
            # Didn't find space so start inference with just the left context tokens
            truncated_tokens = self.left_context_tokens

        if self.token_backoff == -1 or len(tokens) - self.token_backoff < len(truncated_tokens):
            tokens = truncated_tokens
        else:
            tokens = tokens[:-self.token_backoff]

        # Build up the sequence text for the context
        # Start after the left context tokens
        sequence_text = ""
        for i in range(len(self.left_context_tokens), len(tokens)):
            sequence_text = sequence_text + self.index_to_word_lower[tokens[i]]
        valid = [(0.0, tokens, sequence_text)]
        heapq.heapify(valid)

        # Create a hash mapping each valid following character to a list of log probabilities
        char_to_log_probs = {}

        while len(valid) > 0:
            # Only work on the top hypotheses from the last round of extension
            current = list(valid)

            # Add new extended hypotheses to this list
            valid.clear()

            # Keep going until we have extended all hypotheses in the current set
            while len(current) > 0:
                current_batch = 0
                batch_tensors = []
                batch_sequences = []
                batch_likelihoods = []
                batch_seq_text = []
                while len(current) > 0 and current_batch < self.batch_size:
                    # Get the new sequence to work on
                    (current_likelihood, sequence, sequence_text) = current.pop(0)
                    batch_tensors += torch.tensor(sequence),
                    batch_sequences += sequence,
                    batch_likelihoods += current_likelihood,
                    batch_seq_text += sequence_text,
                    current_batch += 1

                tokens_tensor = torch.stack(tuple(batch_tensors)).to(self.device)

                with torch.no_grad():
                    logits = self.model(tokens_tensor).logits
                    log_probs = torch.log_softmax(logits[:, -1, :], dim=1).to("cpu")

                for j in range(current_batch):
                    sequence_text = batch_seq_text[j]
                    vocab = []
                    extra_vocab = []

                    remaining_context = converted_context_lower[len(sequence_text):]
                    if len(remaining_context) == 0:
                        vocab = self.valid_vocab
                    else:
                        if remaining_context in self.vocab:
                            vocab = self.vocab[remaining_context]
                        for i in range(1, len(remaining_context)):
                            tokenization = self._encode(context_lower[len(sequence_text):len(sequence_text) + i])
                            if len(tokenization) == 1:
                                extra_vocab += tokenization[0],

                    # Create a list of token indexes that are a prefix of target text
                    # We go over all the integer IDs in the vocab and extra_vocab lists
                    for i in itertools.chain(vocab, extra_vocab):
                        hypo_str = sequence_text + self.index_to_word_lower[i]
                        hypo_seq = batch_sequences[j].copy()
                        hypo_seq += i,

                        # Add the log prob of this token to the previous running total
                        # For some reason the float cast makes it run faster
                        likelihood = batch_likelihoods[j] + float(log_probs[j][i])
                        # If we have extended to a space following the context, then that hypothesis gets to be done
                        # This takes a lot longer that just requiring extending beyond existing context
                        # Just require hypotheses to extend beyond the existing typed context
                        if len(hypo_str) > len(context):
                            ch = hypo_str[target_pos]

                            # Create an empty list if we haven't seen this character before
                            if ch not in char_to_log_probs:
                                char_to_log_probs[ch] = []
                            char_to_log_probs[ch] += likelihood,

                        else:
                            hypo = (likelihood, hypo_seq, hypo_str)
                            if len(valid) < self.beam_width:
                                heapq.heappush(valid, hypo)
                            else:
                                heapq.heappushpop(valid, hypo)

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
                char_probs += logsumexp(char_to_log_probs[target_ch]),
            else:
                char_probs += float("-inf"),
        # Normalize to a distribution that sums to 1
        char_probs = softmax(char_probs)

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
        except BaseException:
            raise InvalidLanguageModelException(f"{self.model_name} is not a valid model identifier on HuggingFace.")
        self.vocab_size = self.tokenizer.vocab_size
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        except BaseException:
            raise InvalidLanguageModelException(
                f"{self.model_dir} is not a valid local folder or model identifier on HuggingFace.")

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
