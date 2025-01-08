import torch
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
import heapq

from bcipy.core.symbols import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.exceptions import InvalidLanguageModelException

from scipy.special import logsumexp
from scipy.special import softmax
import time
from collections import defaultdict
from typing import Final
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
                 beam_width: int = None,
                 fp16: bool = False,
                 mixed_case_context: bool = False,
                 case_simple: bool = False,
                 max_completed: int = None,
                 ):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type - SYMBOL only
            symbol_set         - list of symbol strings
            lang_model_name    - name of the Hugging Face casual language model to load
            lm_path            - load fine-tuned model from specified directory
            lm_device          - device to use for making predictions (cpu, mps, or cuda)
            lm_left_context    - text to condition start of sentence on
            beam_width         - how many hypotheses to keep during the search, None=off
            fp16               - convert model to fp16 to save memory/compute on CUDA
            mixed_case_context - use mixed case for language model left context
            case_simple        - simple fixing of left context case
            max_completed      - stop search once we reach this many completed hypotheses, None=don't prune
        """
        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.valid_vocab = []
        self.vocab = defaultdict(list)
        # Since subword token ids are integers, use a list instead of a
        # dictionary
        self.index_to_word = []
        self.index_to_word_lower = []
        self.symbol_set_lower = None
        self.device = lm_device
        self.left_context = lm_left_context
        self.fp16 = fp16
        self.mixed_case_context = mixed_case_context
        self.case_simple = case_simple
        self.max_completed = max_completed

        if not max_completed and not beam_width:
            print(
                f"WARNING: using causal language model without any pruning, this can be slow!")
        else:
            print(f"Causal language model, beam_width {
                  beam_width}, max_completed {max_completed}")

        # We optionally load the model from a local directory, but if this is not
        # specified, we load a Hugging Face model

        causal_params = self.parameters['causal']
        self.model_name = lang_model_name or causal_params['model_name']['value']

        local_model_path = lm_path or causal_params['model_path']['value']
        self.model_dir = f"{
            LM_PATH}/{local_model_path}" if local_model_path != "" else self.model_name

        # Parameters for the search
        self.beam_width = beam_width

        # Simple heuristic to correct case in the LM context
        self.simple_upper_words = {"i": "I",
                                   "i'll": "I'll",
                                   "i've": "I've",
                                   "i'd": "I'd",
                                   "i'm": "I'm"}

        # Track how much time spent in different parts of the predict function
        self.predict_total_ns = 0
        self.predict_inference_ns = 0

        # Are we a model that automatically inserts a start token that we need
        # to get rid of
        self.drop_first_token = False

        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    def _build_vocab(self) -> None:
        """
        Build a vocabulary table mapping token index to word strings
        """

        # Loop over all the subword tokens in the LLM
        for i in range(self.vocab_size):
            # Create a map from the subword token integer ID to the mixed and
            # lowercase string versions
            word = self.tokenizer.decode([i])
            word_lower = word.lower()
            self.index_to_word += word,
            self.index_to_word_lower += word_lower,

            # Check if all the characters in the subword token are in our valid
            # symbol set
            valid = True
            for ch in word_lower:
                # The space char is only valid once we convert spaces to the
                # space char
                if ch == SPACE_CHAR:
                    valid = False
                    break
                if ch == ' ':
                    continue
                elif ch not in self.symbol_set_lower:
                    valid = False
                    break

            # If the subword token symbols are all valid, then add it to the
            # list of valid token IDs
            if valid:
                self.valid_vocab += i,
                # Add this token ID to all lists for its valid text prefixes
                for j in range(len(word)):
                    key = word_lower[0:j + 1].replace(' ', SPACE_CHAR)
                    self.vocab[key] += i,

        # When done, self.vocab can be used to map to possible following subword tokens given some text, e.g.:
        # self.vocab["cyclo"] = [47495, 49484]
        # self.index_to_word[self.vocab["cyclo"][0]] = cyclop
        # self.index_to_word[self.vocab["cyclo"][1]] = cyclopedia

        (self.model_name.startswith("facebook/opt")
         or self.model_name.startswith("figmtu/opt")
         or "Llama-3.1" in self.model_name)

        # Get the index we use for the start or end pseudo-word
        if self.left_context == "":
            if "gpt2" in self.model_name:
                self.left_context = "<|endoftext|>"
            elif "Llama-3.1" in self.model_name:
                self.left_context = "<|begin_of_text|>"
            # Seems to have both sentence start and end tokens:
            # https://docs.mistral.ai/guides/tokenization/
            elif "Mistral" in self.model_name:
                self.left_context = "<s>"
            else:
                self.left_context = "</s>"

        # OPT, Llama and Mistral all insert start token
        self.drop_first_token = (self.model_name.startswith("facebook/opt") or
                                 self.model_name.startswith("figmtu/opt") or
                                 "Llama-3.1" in self.model_name or
                                 "Mistral" in self.model_name)

        # Get token id(s) for the left context we condition all sentences on
        self.left_context_tokens = self._encode(self.left_context)
        print(
            f"Causal: left_context = '{
                self.left_context}', left_context_tokens = {
                self.left_context_tokens}")

    def _encode(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        # Both OPT and Llama automatically insert a start token which we want
        # to control ourselves
        if len(tokens) > 1 and self.drop_first_token:
            tokens = tokens[1:]

        return tokens

    def _sequence_string(self, sequence: List[int]) -> str:
        """
        Convert a sequence of subword token IDs into a string with each token in ()'s
        :param sequence: List of subword token IDs
        :return: String
        """
        return "".join([f"({self.index_to_word[x]})" for x in sequence])

    def get_all_tokens_text(self):
        """
        Return an array with the text of all subword tokens.
        The array is in order by the integer index into the vocabulary.
        This is mostly just for exploring the tokens in different LLMs.
        :return: Array of subword token text strings.
        """
        result = []
        for i in range(self.vocab_size):
            result.append(self.tokenizer.decode([i]))
        return result

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
        start_ns = time.time_ns()

        converted_context = "".join(evidence)
        converted_context_lower = converted_context.lower()
        context = converted_context.replace(SPACE_CHAR, ' ')

        # If using the simple case feature, we need to go through the actual
        # left context and capitalize the first letter in the sentence as
        # well as any word in our list of words that should be capitalized.
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
            context = cased_context

        context_lower = context.lower()

        # Index in the hypothesis string that is the next character after our
        # context
        target_pos = len(context_lower)

        # For stats purposes track length of the prefix we are extending from space to match
        # prefix_len = target_pos

        # Look for the last space in the context, or -1 if no begin_text in
        # context yet
        pos = context_lower.rfind(" ")
        tokens = []
        tokens.extend(self.left_context_tokens)
        if pos >= 0:
            # Optionally, we condition on upper and lower case left context
            if self.mixed_case_context:
                truncated_context = context[0:pos]
            else:
                truncated_context = context_lower[0:pos]
            tokens.extend(self._encode(truncated_context))
            # prefix_len -= pos

        # print(f"DEBUG, {context_lower} pos {pos}, prefix_len {prefix_len}")

        # Constant indexes for use with the hypotheses tuples
        LOGP: Final[int] = 0
        SEQ: Final[int] = 1
        LEN: Final[int] = 2

        # Our starting hypothesis that we'll be extending.
        # Format is (log likelihood, token id sequence, text length).
        # Note: we only include tokens after any in left context.
        start_length = 0
        for x in tokens[len(self.left_context_tokens):]:
            start_length += len(self.index_to_word_lower[x])
        current_hypos = [(0.0, tokens, start_length)]

        # We use a priority queue to track the top hypotheses during the beam search.
        # For a beam of 8, empirical testing showed this was about the same amount
        # of time as a simpler list that used a linear search to replace when
        # full.
        heapq.heapify(current_hypos)

        # Create a hash mapping each valid following character to a list of log
        # probabilities
        char_to_log_probs = defaultdict(list)

        # Add new extended hypotheses to this heap
        next_hypos = []

        # Tracks count of completed hypotheses
        completed = 0

        # Used to signal to while loop to stop the search
        done = False

        # Start a beam search forward from the backed off token sequence.
        # Each iteration of this while loop extends hypotheses by all valid tokens.
        # We only keep at most self.beam_width hypotheses in the valid heap.
        # Stop extending search once we reach our max completed target.
        while len(current_hypos) > 0 and not done:
            # We'll explore hypothesis in order from most probable to least.
            # This has little impact on how long it takes since this is only sorting a small number of things.
            # But it is important with max_completed pruning since we want to
            # bias for completing high probability things.
            current_hypos.sort(reverse=True)

            # Work on the hypotheses from the last round of extension.
            # Create the torch tensor for the inference with a row for each
            # hypothesis.
            tokens_tensor = torch.tensor([x[SEQ] for x in current_hypos]).reshape(
                len(current_hypos), -1).to(self.device)

            before_inference_ns = time.time_ns()
            # Ask the LLM to predict tokens that come after our current set of
            # hypotheses
            with torch.no_grad():
                # Compute the probabilities from the logits
                log_probs = torch.log_softmax(self.model(
                    tokens_tensor).logits[:, -1, :], dim=1)

                # Create a big 2D tensor where each row is that hypothesis' current likelihood.
                # First create a list of just the hypotheses' likelihoods.
                # Then reshape to be a column vector.
                # Then duplicate the column based on the number of subword
                # tokens in the LLM.
                add_tensor = torch.tensor([x[LOGP] for x in current_hypos]).reshape(
                    (log_probs.size()[0], 1)).repeat(1, log_probs.size()[1]).to(self.device)

                # Add the current likelihoods with each subtoken's probability.
                # Move it back to the CPU and convert to numpy since this makes
                # it a lot faster to access for some reason.
                new_log_probs = torch.add(
                    log_probs, add_tensor).detach().cpu().numpy()
            self.predict_inference_ns += time.time_ns() - before_inference_ns

            for current_index, current in enumerate(current_hypos):
                vocab = []
                extra_vocab = []
                # Extending this hypothesis must match the remaining text
                remaining_context = converted_context_lower[current[LEN]:]
                if len(remaining_context) == 0:
                    # There is no remaining context thus all subword tokens that are valid under our symbol set
                    # should be considered when computing the probability of
                    # the next character.
                    vocab = self.valid_vocab
                else:
                    if remaining_context in self.vocab:
                        # We have a list of subword tokens that match the remaining text.
                        # They could be the same length as the remaining text
                        # or longer and have the remaining text as a prefix.
                        vocab = self.vocab[remaining_context]

                    # We may need to use a subword token that doesn't completely consume the remaining text.
                    # Find these by tokenizing all possible lengths of text
                    # starting from the current position.
                    for i in range(1, len(remaining_context)):
                        tokenization = self._encode(
                            context_lower[current[LEN]:current[LEN] + i])
                        # Ignore tokenizations involving multiple tokens since
                        # they involve an ID we would have already added.
                        if len(tokenization) == 1:
                            extra_vocab += tokenization[0],

                # The below code takes the most time, results from pprofile on 5 phrases on an 2080 GPU:
                #    299|  22484582|      89.5763|   3.9839e-06| 14.24%|                for token_id in itertools.chain(vocab, extra_vocab):
                #    300|         0|            0|            0|  0.00%|                    # For a hypothesis to finish it must extend beyond the existing typed context
                #    301|  22483271|      93.7939|  4.17172e-06| 14.91%|                    subword_len = len(self.index_to_word_lower[token_id])
                #    302|  22483271|      92.8608|  4.13022e-06| 14.76%|                    if (current[LEN] + subword_len) > len(context):
                #    303|         0|            0|            0|  0.00%|                        # Add this likelihood to the list for the character at the prediction position.
                #    304|         0|            0|            0|  0.00%|                        # Tracking the list and doing logsumpexp later was faster than doing it for each add.
                #    305|  22480431|      106.353|  4.73094e-06| 16.90%|                        char_to_log_probs[self.index_to_word_lower[token_id][target_pos - current[LEN]]] += new_log_probs[current_index][token_id],
                #    306|  22480431|       92.689|   4.1231e-06| 14.73%|                        completed += 1
                #    307|      2840|    0.0124488|  4.38338e-06|  0.00%|                    elif not self.beam_width or len(next_hypos) <
                #
                # Tuning notes:
                #  - With a beam of 8 and max completed of 32,000, getting around 5x speedup on written dev set.
                #  - This results in a PPL increase of 0.0025 versus old results using only beam of >= 8.
                #  - Pruning based on log probability difference and based on minimum number of hypotheses per symbol in alphabet did worse.
                #  - Code for these other pruning methods was removed.
                # Possible ways to make it faster:
                #  - Stop part way through the below for loop over (vocab, extra_vocab). But this seems weird since the token IDs are in
                #    no particular order, we'd be just stopping early on the last hypothesis being explored by the enclosing loop.
                #  - Sort the rows in the log prob results on the GPU. Use these to limit which token IDs we explore in the below
                # for loop. Is it possible to do this without introducing too
                # much extra work to limit to the high probability ones?

                # Create a list of token indexes that are a prefix of the target text.
                # We go over all the integer IDs in the vocab and extra_vocab
                # lists.
                for token_id in itertools.chain(vocab, extra_vocab):
                    # For a hypothesis to finish it must extend beyond the
                    # existing typed context
                    subword_len = len(self.index_to_word_lower[token_id])
                    if (current[LEN] + subword_len) > len(context):
                        # Add this likelihood to the list for the character at the prediction position.
                        # Tracking the list and doing logsumpexp later was
                        # faster than doing it for each add.
                        char_to_log_probs[self.index_to_word_lower[token_id][target_pos -
                                                                             current[LEN]]] += new_log_probs[current_index][token_id],
                        completed += 1
                    elif not self.beam_width or len(next_hypos) < self.beam_width:
                        # If we are under the beam limit then just add it
                        heapq.heappush(next_hypos,
                                       (new_log_probs[current_index][token_id],
                                        current[SEQ] + [token_id],
                                        current[LEN] + subword_len))
                    elif new_log_probs[current_index][token_id] > next_hypos[0][LOGP]:
                        # Or replace the worst hypotheses with the new one
                        heapq.heappushpop(next_hypos,
                                          (new_log_probs[current_index][token_id],
                                           current[SEQ] + [token_id],
                                           current[LEN] + subword_len))

                # Break out of the for loop over hypotheses and while loop if
                # we reach our max completed goal
                if self.max_completed and completed >= self.max_completed:
                    done = True
                    break

            # Swap in the extended set as the new current working set
            current_hypos = next_hypos
            next_hypos = []

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

        next_char_pred = {}
        for i, ch in enumerate(self.symbol_set_lower):
            if ch is SPACE_CHAR:
                next_char_pred[ch] = char_probs[i]
            else:
                next_char_pred[ch.upper()] = char_probs[i]
        next_char_pred[BACKSPACE_CHAR] = 0.0

        end_ns = time.time_ns()
        self.predict_total_ns += end_ns - start_ns

        return list(sorted(next_char_pred.items(),
                    key=lambda item: item[1], reverse=True))

    def dump_predict_times(self) -> None:
        """Print some stats about the prediction timing"""
        if self.predict_total_ns > 0:
            print(f"Predict %: "
                  f"inference {self.predict_inference_ns / self.predict_total_ns * 100.0:.3f}")

    def update(self) -> None:
        """Update the model state"""
        ...

    def load(self) -> None:
        """
            Load the language model and tokenizer, initialize class variables
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=False)
        except BaseException:
            raise InvalidLanguageModelException(
                f"{self.model_name} is not a valid model identifier on HuggingFace.")
        self.vocab_size = self.tokenizer.vocab_size
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
            if self.fp16 and self.device == "cuda":
                self.model = self.model.half()
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

    def get_num_parameters(self) -> int:
        """
            Find out how many parameters the loaded model has
        Args:
        Response:
            Integer number of parameters in the transformer model
        """
        return sum(p.numel() for p in self.model.parameters())

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
