from collections import Counter
import torch
from typing import Dict, List, Tuple
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from timeit import default_timer as timer

from bcipy.helpers.task import BACKSPACE_CHAR, SPACE_CHAR, alphabet
from bcipy.language.main import LanguageModel, ResponseType

from scipy.special import logsumexp
from scipy.special import softmax

class CausalLanguageModel(LanguageModel):
    """Character language model based on a pre-trained causal model, GPT-2 by default."""

    def __init__(self, response_type: ResponseType, symbol_set: List[str], lm_path: str = None):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type - SYMBOL only
            symbol_set - list of symbol strings
            lm_path - path to language model files
        """
        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.valid_vocab = []
        self.index_to_word = {}
        self.index_to_word_lower = {}
        self.start_end_index = None
        self.space_index = None
        self.lm_path = lm_path or "gpt2"
        self.symbol_set_lower = None

        # parameters for search
        self.beam_width = 8
        self.batch_size = 8
        self.token_backoff = -1

        # self.search_depth = 2

        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    def __build_vocab(self) -> None:
        """
        Build a vocabulary table mapping token index to word strings
        """

        for i in range(self.vocab_size):
            word = self.tokenizer.decode([i])
            self.index_to_word[i] = word
            self.index_to_word_lower[i] = word.lower()
            valid = True
            for ch in word.lower():
                if ch.isspace():
                    ch = SPACE_CHAR
                if ch not in self.symbol_set_lower:
                    valid = False
                    break
            if valid:
                self.valid_vocab.append(i)

        # Get the index we use for the start or end pseudo-word
        self.start_end_index = self.tokenizer.encode("<|endoftext|>")[0]

        # Index of the space character
        self.space_index = self.tokenizer.encode(" ")[0]

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

        context = "".join(evidence)

        context = context.replace(SPACE_CHAR, ' ')
        context_lower = context.lower()
        
        # Index in the hypothesis string that is the next character after our context
        target_pos = len(context)

        # Remove token_backoff tokens from the end of the context
        # If token_backoff is -1 or goes beyond a word boundary, then
        # search from the last space character in the context
        # If no space, then from the very beginning
        valid = []
        truncated_tokens = []
        tokens = self.tokenizer.encode(context)
        tokens.insert(0, self.space_index)

        pos = context.rfind(" ")
        if pos >= 0:
            truncated_context = context[0:pos]
            truncated_tokens = self.tokenizer.encode(truncated_context)
            truncated_tokens.insert(0, self.space_index)
        else:
            truncated_tokens = [self.space_index]

        if self.token_backoff == -1 or len(tokens) - self.token_backoff < len(truncated_tokens):
            tokens = truncated_tokens
        else:
            tokens = tokens[:-self.token_backoff]
        valid = [(tokens, 0.0)]


        # Create a hash mapping each valid following character to a list of log probabilities
        char_to_log_probs = {}

        # print(f"Starting search, beam={self.beam_width}, valid={valid}")
        done_best = float("-inf")
        while len(valid) > 0:
            # Only work on the top hypotheses from the last round of extension
            current = sorted(valid, key=lambda x: x[1], reverse=True)
            before = len(current)

            # Popping last element is O(1)
            while len(current) > self.beam_width:
                current.pop(-1)

            # Add new extended hypotheses to this list
            valid = []

            # Keep going until we have extended all hypotheses in the current set
            while len(current) > 0:
                current_batch = 0
                batch_tensors = []
                batch_sequences = []
                batch_likelihoods = []
                while len(current) > 0 and current_batch < self.batch_size:
                    # Get the new sequence to work on
                    (sequence, current_likelihood) = current.pop(0)
                    batch_tensors.append(torch.tensor(sequence).to(self.device))
                    batch_sequences.append(sequence)
                    batch_likelihoods.append(current_likelihood)
                    current_batch += 1

                
                tokens_tensor = torch.stack(tuple(batch_tensors))
                with torch.no_grad():
                    logits = self.model(tokens_tensor).logits
                    log_probs = torch.log(torch.softmax(logits[:, -1, :], dim=1))

                for j in range(current_batch):
                    # Create sequence text before the search sequence, skipping start word, make it all lowercase
                    sequence_text = ""
                    for i in range(1, len(batch_sequences[j])):
                        sequence_text = sequence_text + self.index_to_word_lower[batch_sequences[j][i]]
                    #print(f"sequence_text = '{sequence_text}'")

                    # Create a list of token indexes that are a prefix of target text
                    for i in self.valid_vocab: # range(logits.size()[2]):
                        hypo_str = sequence_text + self.index_to_word_lower[i]
                        #print(f"hypo_str = '{hypo_str}'")

                        # In some cases hypothesis is shorter than context, in some cases longer
                        if hypo_str.startswith(context_lower) or context_lower.startswith(hypo_str):
                            # print(f"hypo_str = '{hypo_str}', {i}: '{self.index_to_word[i]}' {predictions[-1, i]:.4f}")
                            # If we are also the same length, then we must be the same string (sans case)
                            hypo_seq = batch_sequences[j].copy()
                            hypo_seq.append(i)

                            # Add the log prob of this token to the previous running total
                            likelihood = batch_likelihoods[j] + float(log_probs[j][i])

                            # If we have extended to a space following the context, then that hypothesis gets to be done
                            # This takes a lot longer that just requiring extending beyond existing context
                            #last_space_pos = hypo_str.rfind(" ")
                            #if last_space_pos >= len(context):
                            # Just require hypotheses to extend beyond the existing typed context
                            if len(hypo_str) > len(context):
                                # Track the most probable finishing hypothesis
                                # if likelihood > done_best:
                                #     done_best = likelihood

                                hypo_str = ""
                                # Note: Skipping index 0 since this is the space character we forced at the start
                                for i in range(1, len(hypo_seq)):
                                    hypo_str = hypo_str + self.index_to_word_lower[hypo_seq[i]]
                                #print(f"hypo_str = '{hypo_str}'")
                                ch = hypo_str[target_pos]

                                # Map any type of following whitespace character to be our space symbol
                                if ch.isspace():
                                    ch = SPACE_CHAR

                                # Create an empty list if we haven't seen this character before
                                if ch not in char_to_log_probs:
                                    char_to_log_probs[ch] = []
                                char_to_log_probs[ch].append(likelihood)

                            else:
                                hypo = (hypo_seq, likelihood)
                                valid.append(hypo)

        # Parallel array to symbol_set for storing the marginals
        char_probs = []
        for ch in self.symbol_set_lower:
            # Handle cases when symbols are never seen
            if ch in char_to_log_probs:
                char_probs.append(logsumexp(char_to_log_probs[ch]))
            else:
                char_probs.append(float("-inf"))
        # Normalize to a distribution that sums to 1
        char_probs = softmax(char_probs)

        next_char_pred = Counter()

        for i, ch in enumerate(self.symbol_set_lower):
            if ch is SPACE_CHAR:
                next_char_pred[ch] = char_probs[i]
            else:
                next_char_pred[ch.upper()] = char_probs[i]
        
        next_char_pred[BACKSPACE_CHAR] = 0.0

        # width = 120
        # print("             " + "_" * width)
        # for i in range(len(self.symbol_set_lower)):
        #     print(f"{self.symbol_set_lower[i]} = {char_probs[i]:4.2e} " + "*" * int(char_probs[i] * width))
        # print("             " + "_" * width)

        return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))

    def update(self) -> None:
        """Update the model state"""
        ...

    def load(self) -> None:
        """
            Load the language model and tokenizer, initialize class variables
        """
        self.model = GPT2LMHeadModel.from_pretrained(self.lm_path)
        self.model.eval()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.lm_path)
        self.vocab_size = self.tokenizer.vocab_size
        
        # If you have a GPU, put everything on cuda
        self.device = "cpu"
        # self.device = "cuda"   # NVidia GPU
        # self.device = "mps"    # M1 mac
        self.model.to(self.device)

        self.symbol_set_lower = []
        for ch in self.symbol_set:
            if ch is SPACE_CHAR:
                self.symbol_set_lower.append(SPACE_CHAR)
            elif ch is BACKSPACE_CHAR:
                continue
            else:
                self.symbol_set_lower.append(ch.lower())
                
        self.__build_vocab()

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
