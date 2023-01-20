from collections import Counter
import torch
from typing import Dict, List, Tuple
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from timeit import default_timer as timer
import heapq

from bcipy.language.main import BACKSPACE_CHAR, SPACE_CHAR, alphabet
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
        self.vocab = {}
        self.longest_token = 0
        self.index_to_word = {}
        self.index_to_word_lower = {}
        self.start_end_index = None
        self.space_index = None
        self.lm_path = lm_path or "gpt2"
        self.symbol_set_lower = None

        # parameters for search
        self.beam_width = 8
        self.batch_size = 8

        # backoff to the previous space
        self.token_backoff = -1

        # self.search_depth = 2

        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    def __convert_space(self, s: str) -> str:
        ret = ""
        for ch in s:
            if ch == ' ':
                ret += SPACE_CHAR
            else:
                ret += ch
        return ret

    def __build_vocab(self) -> None:
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
                    key = self.__convert_space(word_lower[0:j+1])
                    if key not in self.vocab:
                        self.vocab[key] = []
                    self.vocab[key] += i,

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

        converted_context = "".join(evidence)

        context = converted_context.replace(SPACE_CHAR, ' ')
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

        # Build up the sequence text for the context
        sequence_text = ""
        for i in range(1, len(tokens)):
            sequence_text = sequence_text + self.index_to_word_lower[tokens[i]]
        valid = [(0.0, tokens, sequence_text)]

        heapq.heapify(valid)

        # Create a hash mapping each valid following character to a list of log probabilities
        char_to_log_probs = {}

        # print(f"Starting search, beam={self.beam_width}, valid={valid}")
        done_best = float("-inf")
        while len(valid) > 0:
            # Only work on the top hypotheses from the last round of extension
            # current = sorted(valid, key=lambda x: x[1], reverse=True)
            current = list(valid)
            before = len(current)

            # Popping last element is O(1)
            # while len(current) > self.beam_width:
            #     current.pop(-1)

            # Add new extended hypotheses to this list
            valid.clear()
            # valid = []

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
                    batch_tensors += torch.tensor(sequence).to(self.device),
                    batch_sequences += sequence,
                    batch_likelihoods += current_likelihood,
                    batch_seq_text += sequence_text,
                    current_batch += 1
                
                tokens_tensor = torch.stack(tuple(batch_tensors))
                with torch.no_grad():
                    logits = self.model(tokens_tensor).logits
                    log_probs = torch.log(torch.softmax(logits[:, -1, :], dim=1))

                for j in range(current_batch):
                    sequence_text = batch_seq_text[j]
                    # for i in range(1, len(batch_sequences[j])):
                    #     sequence_text = sequence_text + self.index_to_word_lower[batch_sequences[j][i]]
                    #print(f"sequence_text = '{sequence_text}'")

                    vocab = []

                    remaining_context = converted_context[len(sequence_text):]
                    if len(remaining_context) == 0:
                        vocab = self.valid_vocab
                    else:
                        if remaining_context in self.vocab:
                            vocab = self.vocab[remaining_context]
                        for i in range(1,len(remaining_context)):
                            tokenization = self.tokenizer.encode(context[len(sequence_text):len(sequence_text)+i])
                            if len(tokenization) == 1:
                                vocab += tokenization[0],


                    # Create a list of token indexes that are a prefix of target text
                    for i in vocab: # range(logits.size()[2]):
                        hypo_str = sequence_text + self.index_to_word_lower[i]
                        # print(f"hypo_str = '{hypo_str}', {i}: '{self.index_to_word[i]}' {predictions[-1, i]:.4f}")
                        
                        hypo_seq = batch_sequences[j].copy()
                        hypo_seq += i,

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

                            #TODO why do we rebuild hypo_str?
                            # hypo_str = batch_seq_text[j] + self.index_to_word_lower[hypo_seq[-1]]
                            # Note: Skipping index 0 since this is the space character we forced at the start
                            # for i in range(1, len(hypo_seq)):
                            #     hypo_str = hypo_str + self.index_to_word_lower[hypo_seq[i]]
                            #print(f"hypo_str = '{hypo_str}'")
                            ch = hypo_str[target_pos]

                            # Map a space character to be our space symbol
                            ch = self.__convert_space(ch)

                            # Create an empty list if we haven't seen this character before
                            if ch not in char_to_log_probs:
                                char_to_log_probs[ch] = []
                            char_to_log_probs[ch] += likelihood,

                        else:
                            hypo = (likelihood, hypo_seq, hypo_str)
                            # print(f"Pushed: {hypo}\n")
                            if len(valid) < self.beam_width:
                                heapq.heappush(valid, hypo)
                                # valid.append(hypo)
                            else:
                                popped = heapq.heappushpop(valid, hypo)
                                # print(f"Popped: {popped}\n")

        # Parallel array to symbol_set for storing the marginals
        char_probs = []
        for ch in self.symbol_set_lower:
            # Handle cases when symbols are never seen
            if ch in char_to_log_probs:
                char_probs += logsumexp(char_to_log_probs[ch]),
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
