import torch
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
import heapq
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.helpers.symbols import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.helpers.exceptions import InvalidLanguageModelException
from scipy.special import logsumexp
from scipy.special import softmax
import time
from collections import defaultdict
from typing import Final
import re
from bcipy.config import LM_PATH

class AmbiguousLanguageModel(LanguageModel):
    """Word language model based on a pre-trained causal model"""

    def __init__(self,
                 response_type: ResponseType,
                 symbol_set: List[str],
                 lang_model_name: Optional[str] = None,
                 lm_path: Optional[str] = None,
                 lm_device: str = "cpu",
                 lm_left_context: str = "",
                 beam_width: int = None,
                 fp16: bool = True,
                 mixed_case_context: bool = True,
                 case_simple: bool = True,
                 max_completed: int = 30,
                 completions: bool = False
                 ):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type      - WORD only
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
            completions        - whether to offer suggestions longer than the input group sequence
        """
        assert response_type is ResponseType.WORD, "Ambiguous response type must be word."
        
        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.valid_vocab = []

        # Key: stringified group sequence, Value: list of tokens that exactly match 
        self.vocab_exact = defaultdict(list)
        # Key: stringified group sequence, Value: list of tokens that exactly match but start with a space
        self.vocab_exact_space = defaultdict(list)
        # Key: stringified group sequence, Value: list of tokens that exactly match or start with the full group sequence
        self.vocab_prefix = defaultdict(list)
        # Key: stringified group sequence, Value: list of tokens that exactly match or start with the full group sequence, token starts with space
        self.vocab_prefix_space = defaultdict(list)
        
        # Since subword token ids are integers, use a list instead of a dictionary
        self.index_to_word = []
        self.index_to_word_lower = []
        self.symbol_set_lower = None
        self.device = lm_device
        self.left_context = lm_left_context
        self.fp16 = fp16
        self.mixed_case_context = mixed_case_context
        self.case_simple = case_simple
        self.max_completed = max_completed
        self.completions = completions

        if not max_completed and not beam_width:
            print(f"WARNING: using ambiguous language model without any pruning, this can be slow!")
        else:
            print(f"Ambiguous language model, beam_width {beam_width}, max_completed {max_completed}")

        # We optionally load the model from a local directory, but if this is not
        # specified, we load a Hugging Face model

        ambig_params = self.parameters['ambiguous']
        self.model_name = lang_model_name or ambig_params['model_name']['value']

        local_model_path = lm_path or ambig_params['model_path']['value']
        self.model_dir = f"{LM_PATH}/{local_model_path}" if local_model_path != "" else self.model_name

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


        # Are we a model that automatically inserts a start token that we need to get rid of
        self.drop_first_token = False

        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.WORD]
    
    # Takes in text and returns a string representation of ambiguous groups
    # Returns None if non-alpha characters
    def _ambiguate_word(self, text: str) -> str:
        groups = ''
        for ch in text:
            group = [a for a, b in self.character_groups.items() if ch in b]
            if len(group) == 1:
                groups += str(group[0])
            else:
                return None
        return groups
        
    def _build_vocab(self) -> None:
        """
        Build a vocabulary table mapping token index to word strings
        """
        self.character_groups = {1: ['a','b','c','d','e'], 2: ['f','g','h','i','j','k','l','m'], 3: ['n','o','p','q','r'], 4: ['s','t','u','v','w','x','y','z']}

        # print('Building vocab...')
        
        # Loop over all the subword tokens in the LLM
        for i in range(self.vocab_size):
            # Create a map from the subword token integer ID to the mixed and lowercase string versions
            word = self.tokenizer.decode([i])
            word_lower = word.lower()
            self.index_to_word += word,
            self.index_to_word_lower += word_lower,
            space_start = False
            
            if word_lower[0] == ' ':
                word_lower = word_lower[1:]
                space_start = True
            
            groups = self._ambiguate_word(word_lower)

#            print(f'Token {i}, \"{word}\", Lower: {word_lower}, Ambiguation: {groups}')
            
            # If the subword token symbols are all valid, then add it to the list of valid token IDs
            if groups:
                self.valid_vocab += i,

                if space_start:
                    self.vocab_exact_space[groups] += i,
                else:
                    # Add this token ID to list for exact group match
                    self.vocab_exact[groups] += i,
                
                # Add this token ID to all lists for its valid group prefixes
                # Include exact group match, for word completions, use vocab_prefix INSTEAD of vocab_exact
                for j in range(len(groups)):
                    key = groups[0:j + 1]
                    if space_start:
                        self.vocab_prefix_space[key] += i,
                    else:
                        self.vocab_prefix[key] += i,

        # When done, self.vocab can be used to map to possible following subword tokens given some text, e.g.:
        # self.vocab["cyclo"] = [47495, 49484]
        # self.index_to_word[self.vocab["cyclo"][0]] = cyclop
        # self.index_to_word[self.vocab["cyclo"][1]] = cyclopedia

        (self.model_name.startswith("facebook/opt") or self.model_name.startswith("figmtu/opt") or "Llama-3.1" in self.model_name)

        # Get the index we use for the start or end pseudo-word
        if self.left_context == "":
            if "gpt2" in self.model_name:
                self.left_context = "<|endoftext|>"
            elif "Llama-3.1" in self.model_name:
                self.left_context = "<|begin_of_text|>"
            # Seems to have both sentence start and end tokens: https://docs.mistral.ai/guides/tokenization/
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
        print(f"Causal: left_context = '{self.left_context}', left_context_tokens = {self.left_context_tokens}")

    def _encode(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        # Both OPT and Llama automatically insert a start token which we want to control ourselves
        if self.drop_first_token:
            if len(tokens) > 1:
                tokens = tokens[1:]
            else:
                tokens = []

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

    def predict(self, evidence: List[str], groups: List[int]) -> List[Tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of
        the next symbol
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbols with probability
        """

        assert self.model is not None, "language model does not exist!"

        if not len(groups) and not self.completions:
            return []
        
        start_ns = time.time_ns()

        groups_str = "".join([str(x) for x in groups])        

        group_pattern = r'[1-4]*$'
        assert re.match(group_pattern, groups_str) is not None, "invalid group provided!"

        ending_space = False
        
        context = "".join(evidence)
        context = context.replace(SPACE_CHAR, ' ')

        # Gobble trailing space, but set flag so we know tokens should start with a space
        if len(context) and context[-1] == " ":
            context = context[:-1]
            ending_space = True

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
            context = cased_context

        context_lower = context.lower()

        tokens = []
        tokens.extend(self.left_context_tokens)
        if self.mixed_case_context:
            cased_context = context
        else:
            cased_context = context_lower
        tokens.extend(self._encode(cased_context))
        
        # print(f"DEBUG, {context_lower} pos {pos}, prefix_len {prefix_len}")

        # Constant indexes for use with the hypotheses tuples
        LOGP: Final[int] = 0
        SEQ: Final[int] = 1
        LEN: Final[int] = 2

        # Our starting hypothesis that we'll be extending.
        # Format is (log likelihood, token id sequence, hypothesis length).
        # Hypothesis length is the character-length of the prediction - not including any context
        hypo_length = 0
        current_hypos = [(0.0, tokens, hypo_length)]

        # We use a priority queue to track the top hypotheses during the beam search.
        # For a beam of 8, empirical testing showed this was about the same amount
        # of time as a simpler list that used a linear search to replace when full.
        heapq.heapify(current_hypos)

        # Add new extended hypotheses to this heap
        next_hypos = []

        # Tracks log probs of compelted hypotheses. Due to casing and tokenizations we might get dupes
        word_to_log_probs = defaultdict(list)

        # How many hypotheses have we finished?
        # completed = 0

        # Used to signal to while loop to stop the search
        done = False

        # Start a beam search forward from the backed off token sequence.
        # Each iteration of this while loop extends hypotheses by all valid tokens.
        # We only keep at most self.beam_width hypotheses in the valid heap.
        # Stop extending search once we reach our max completed target.
        while len(current_hypos) > 0 and not done:
            # We'll explore hypothesis in order from most probable to least.
            # This has little impact on how long it takes since this is only sorting a small number of things.
            # But it is important with max_completed pruning since we want to bias for completing high probability things.
            current_hypos.sort(reverse=True)

            # for hypo in current_hypos:
            #     print(str(hypo) + ' ' + self.tokenizer.decode(hypo[SEQ]))

            # Work on the hypotheses from the last round of extension.
            # Create the torch tensor for the inference with a row for each hypothesis.
            tokens_tensor = torch.tensor([x[SEQ] for x in current_hypos]).reshape(len(current_hypos), -1).to(self.device)

            before_inference_ns = time.time_ns()
            # Ask the LLM to predict tokens that come after our current set of hypotheses
            with torch.no_grad():
                # Compute the probabilities from the logits
                log_probs = torch.log_softmax(self.model(tokens_tensor).logits[:, -1, :], dim=1)

                # Create a big 2D tensor where each row is that hypothesis' current likelihood.
                # First create a list of just the hypotheses' likelihoods.
                # Then reshape to be a column vector.
                # Then duplicate the column based on the number of subword tokens in the LLM.
                add_tensor = torch.tensor([x[LOGP] for x in current_hypos]).reshape((log_probs.size()[0], 1)).repeat(1, log_probs.size()[1]).to(self.device)

                # Add the current likelihoods with each subtoken's probability.
                # Move it back to the CPU and convert to numpy since this makes it a lot faster to access for some reason.
                new_log_probs = torch.add(log_probs, add_tensor).detach().cpu().numpy()
            self.predict_inference_ns += time.time_ns() - before_inference_ns

            for current_index, current in enumerate(current_hypos):
                # Create a list of token indexes that are a prefix of the target text.
                vocab = []
                # Extending this hypothesis must match the remaining input groups
                remaining_groups = groups_str[current[LEN]:]
                if len(remaining_groups) == 0:
                    if self.completions:
                        # There is no remaining context thus all subword tokens that are valid under our symbol set
                        # should be considered when computing the probability of the next character.
                        vocab = self.valid_vocab
                        print("Full valid vocab")
                    else:
                        # This prediction is done, should not be in this list
                        print("ERROR: This prediction should have been completed and removed from hypotheses")
                        continue
                else:
                    for i in range(1, len(remaining_groups)):
                        group_prefix = remaining_groups[:i]
                        print(f"remaining groups {remaining_groups}, group prefix {group_prefix}")
                        # We may need to use a subword token that doesn't completely consume the remaining text.
                        # Add tokens that are an exact group match for the prefix of reamining input groups
                        # e.g. "te" for 4,1,4,4
                        # Do not include full group sequence in this, only proper prefixes
                        if current[LEN] == 0 and ending_space:
                            if group_prefix in self.vocab_exact_space:
                                vocab += self.vocab_exact_space[group_prefix]
                                toks = self.vocab_exact_space[group_prefix]
                                for tok in toks:
                                    print(f"Adding token {tok}, \"{self.index_to_word[tok]}\"")
                        else:
                            if group_prefix in self.vocab_exact:
                                vocab += self.vocab_exact[group_prefix]
                                toks = self.vocab_exact[group_prefix]
                                for tok in toks:
                                    print(f"Adding token {tok}, \"{self.index_to_word[tok]}\"")

                    # If completions are allowed, consider tokens that start with the remaining group sequence
                    # This includes exact match
                    if self.completions:
                        if current[LEN] == 0 and ending_space:
                            vocab += self.vocab_prefix_space[remaining_groups]
                            toks = self.vocab_prefix_space[remaining_groups]
                            for tok in toks:
                                print(f"Adding token {tok}, \"{self.index_to_word[tok]}\"")
                        else:
                            vocab += self.vocab_prefix[remaining_groups]
                            toks = self.vocab_prefix[remaining_groups]
                            for tok in toks:
                                print(f"Adding token {tok}, \"{self.index_to_word[tok]}\"")
                    # Otherwise, only consider exact match
                    else:
                        if current[LEN] == 0 and ending_space:
                            if remaining_groups in self.vocab_exact_space:
                                vocab += self.vocab_exact_space[remaining_groups]
                                toks = self.vocab_exact_space[remaining_groups]
                                for tok in toks:
                                    print(f"Adding token {tok}, \"{self.index_to_word[tok]}\"")
                        else:
                            if remaining_groups in self.vocab_exact:
                                vocab += self.vocab_exact[remaining_groups]
                                toks = self.vocab_exact[remaining_groups]
                                for tok in toks:
                                    print(f"Adding token {tok}, \"{self.index_to_word[tok]}\"")

                # If we are doing completions, we'll need this 
                word = self.tokenizer.decode(current[SEQ][len(tokens):]).lower()

                # We go over all the integer IDs in the vocab list.
                for token_id in vocab:
                    if self.completions:
                        # A hypothesis finishes if EXISTING hypo meets or exceeds group count
                        # Probability of hypo is summed over all next tokens starting with space
                        token_len = len(self.index_to_word_lower[token_id].replace(" ", ""))
                        if (current[LEN]) >= len(groups):
                            if self.index_to_word_lower[token_id][0] == " ":
                                # Add this likelihood to the list of completed predictions 
                                word_to_log_probs[word] += new_log_probs[current_index][token_id],
                            # print(f'Prediction completed: {word}, {new_log_probs[current_index][token_id]}')
                            elif not self.beam_width or len(next_hypos) < self.beam_width:
                                # If we are under the beam limit then just add it
                                heapq.heappush(next_hypos,
                                            (new_log_probs[current_index][token_id],
                                                current[SEQ] + [token_id],
                                                current[LEN] + token_len))
                                
                            elif new_log_probs[current_index][token_id] > next_hypos[0][LOGP]:
                                # Or replace the worst hypotheses with the new one
                                heapq.heappushpop(next_hypos,
                                                (new_log_probs[current_index][token_id],
                                                current[SEQ] + [token_id],
                                                current[LEN] + token_len))
                        elif not self.beam_width or len(next_hypos) < self.beam_width:
                            # If we are under the beam limit then just add it
                            heapq.heappush(next_hypos,
                                           (new_log_probs[current_index][token_id],
                                            current[SEQ] + [token_id],
                                            current[LEN] + token_len))
                            
                        elif new_log_probs[current_index][token_id] > next_hypos[0][LOGP]:
                            # Or replace the worst hypotheses with the new one
                            heapq.heappushpop(next_hypos,
                                              (new_log_probs[current_index][token_id],
                                               current[SEQ] + [token_id],
                                               current[LEN] + token_len))
                    else:
                        # For a hypothesis to finish it must exactly match the input groups length
                        token_len = len(self.index_to_word_lower[token_id].replace(" ", ""))
                        if (current[LEN] + token_len) == len(groups):
                            # Add this likelihood to the list of completed predictions 
                            word = self.tokenizer.decode(current[SEQ][len(tokens):]).lower() + self.index_to_word_lower[token_id]
                            word_to_log_probs[word] += new_log_probs[current_index][token_id],
                            # print(f'Prediction completed: {word}, {new_log_probs[current_index][token_id]}')
                        elif not self.beam_width or len(next_hypos) < self.beam_width:
                            # If we are under the beam limit then just add it
                            heapq.heappush(next_hypos,
                                           (new_log_probs[current_index][token_id],
                                            current[SEQ] + [token_id],
                                            current[LEN] + token_len))
                            
                        elif new_log_probs[current_index][token_id] > next_hypos[0][LOGP]:
                            # Or replace the worst hypotheses with the new one
                            heapq.heappushpop(next_hypos,
                                              (new_log_probs[current_index][token_id],
                                               current[SEQ] + [token_id],
                                               current[LEN] + token_len))

                # Break out of the for loop over hypotheses and while loop if we reach our max completed goal
                if self.max_completed and len(word_to_log_probs.items()) >= self.max_completed:
                    done = True
                    break

            # Swap in the extended set as the new current working set
            current_hypos = next_hypos
            next_hypos = []

        # Word dict for storing final predictions
        predictions = {}
        for word in word_to_log_probs:
            predictions[word.upper()] = logsumexp(word_to_log_probs[word])
          
        end_ns = time.time_ns()
        self.predict_total_ns += end_ns - start_ns

        # print(f'Total completed predictions: {completed}')
        
        return list(sorted(predictions.items(), key=lambda item: item[1], reverse=True))

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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        except BaseException:
            raise InvalidLanguageModelException(f"{self.model_name} is not a valid model identifier on HuggingFace.")
        self.vocab_size = self.tokenizer.vocab_size
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
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
