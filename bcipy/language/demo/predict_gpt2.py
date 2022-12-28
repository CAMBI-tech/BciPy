# Exploring how to use GPT-2 to make predictions based on context (properly)

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from timeit import default_timer as timer
#from bcipy.helpers.task import alphabet
#from bcipy.helpers.task import BACKSPACE_CHAR, SPACE_CHAR
from scipy.special import logsumexp
from scipy.special import softmax

# Hack to allow running headless
from string import ascii_uppercase
import os

SPACE_CHAR = '_'
BACKSPACE_CHAR = '<'

def alphabet(parameters=None, include_path=True):
    """Alphabet.

    Function used to standardize the symbols we use as alphabet.

    Returns
    -------
        array of letters.
    """
    if parameters and not parameters['is_txt_stim']:
        # construct an array of paths to images
        path = parameters['path_to_presentation_images']
        stimulus_array = []
        for stimulus_filename in sorted(os.listdir(path)):
            # PLUS.png is reserved for the fixation symbol
            if stimulus_filename.endswith(
                    '.png') and not stimulus_filename.endswith('PLUS.png'):
                if include_path:
                    img = os.path.join(path, stimulus_filename)
                else:
                    img = os.path.splitext(stimulus_filename)[0]
                stimulus_array.append(img)
        return stimulus_array

    return list(ascii_uppercase) + [BACKSPACE_CHAR, SPACE_CHAR]

if __name__ == "__main__":
    start = timer()

    # Load the GPT-2 model
    lm_path = "gpt2"               #  117M parameters, 12 layers, d_model  768    ~1.3GB memory during load
    #lm_path = "gpt2-medium"        #  345M             24                 1024    ~3.1GB
    #lm_path = "gpt2-large"         #  762M             36                 1280    ~6.3GB
    #lm_path = "gpt2-xl"            # 1542M             48                 1600   ~12.3GB

    beam_width = 256

    # Previous text context that we are extending
    context = "i prob"

    # Notice how the predictions are much worse without proper case
    #context = "i "
    #context = "I "

    # Long range dependencies, go GPT-2!
    #context = "the space shuttle program was shutdown by n"
    #context = "my cat felix who is very fat, furry, and super cute is having a whole lot of fun chasing a m"

    # Really long last word, search takes a long time!
    #context = "This will be hard to do programmatical"

    # We can now predict the s!
    #context = "i like zebra"

    context_lower = context.lower()

    # Create the symbol set but without backspace
    symbol_set = alphabet()
    symbol_set.remove(BACKSPACE_CHAR)
    # Lowercase since that is what we use for comparisons everywhere
    for i, ch in enumerate(symbol_set):
        symbol_set[i] = ch.lower()

    print(f"Loading {lm_path} model")
    model = GPT2LMHeadModel.from_pretrained(lm_path)
    print(f"Load time = {timer() - start:.2f}")

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()

    print(f"Loading tokenizer")
    tokenizer = GPT2TokenizerFast.from_pretrained(lm_path)
    vocab_size = tokenizer.vocab_size

    # Hash that maps numeric subword token indexes to the actual text
    # Also save a lowercase version for use in case insensitve comparisons
    print(f"Creating index to vocab map, size = {vocab_size}")
    index_to_word = {}
    index_to_word_lower = {}
    for i in range(vocab_size):
        word = tokenizer.decode([i])
        index_to_word[i] = word
        #print(f"{i:6}: '{word}'")
        index_to_word_lower[i] = word.lower()

    # Get the index we use for the start or end pseudo-word
    start_end_index = tokenizer.encode("<|endoftext|>")[0]

    # Index of the space character
    space_index = tokenizer.encode(" ")[0]

    # If you have a GPU, put everything on cuda
    device = "cpu"
    # device = "cuda"   # NVidia GPU
    #device = "mps"    # M1 mac
    model.to(device)

    start_search = timer()

    # We search from the last space character in the context
    # If no space, then from the very beginning
    valid = []
    pos = context.rfind(" ")
    if pos >= 0:
        truncated_context = context[0:pos]
        tokens = tokenizer.encode(truncated_context)
        tokens.insert(0, space_index)
        valid = [(tokens, 0.0)]
        print(f"{tokens}")
        print(f"Searching from truncated context '{truncated_context}'")
    else:
        truncated_context = ""
        valid = [([space_index], 0.0)]
        print(f"Searching from end of text token")

    # List of subword sequences that can produce the text
    done = []

    print(f"Starting search, beam={beam_width}, valid={valid}")
    done_best = float("-inf")
    while len(valid) > 0: # and len(done) < 16000000:
        # Only work on the top hypotheses from the last round of extension
        current = sorted(valid, key=lambda x: x[1], reverse=True)
        before = len(current)
        while len(current) > beam_width:
            current.pop(-1)
        print(f"current, before={before}, after={len(current)}, done={len(done)}, done_best={done_best:.2f}, {current[0]}...{current[-1]}")

        # Add new extended hypotheses to this list
        valid = []

        # Keep going until we have extended all hypotheses in the current set
        while len(current) > 0:
            # Get the new sequence to work on
            (sequence, current_likelihood) = current.pop(0)
            tokens_tensor = torch.tensor(sequence).unsqueeze(0).to(device)
#            t100 = torch.stack((torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device), torch.tensor(sequence).to(device)))

            with torch.no_grad():
                logits = model(tokens_tensor).logits
                log_probs = torch.log(torch.softmax(logits[:, -1, :].flatten(), dim=0))

            # Create sequence text before the search sequence, skipping start word, make it all lowercase
            sequence_text = ""
            for i in range(1, len(sequence)):
                sequence_text = sequence_text + index_to_word_lower[sequence[i]]
            #print(f"sequence_text = '{sequence_text}'")

            # Create a list of token indexes that are a prefix of target text
            for i in range(logits.size()[2]):
                hypo_str = sequence_text + index_to_word_lower[i]
                #print(f"hypo_str = '{hypo_str}'")

                # In some cases hypothesis is shorter than context, in some cases longer
                if hypo_str.startswith(context_lower) or context_lower.startswith(hypo_str):
                    # print(f"hypo_str = '{hypo_str}', {i}: '{index_to_word[i]}' {predictions[-1, i]:.4f}")
                    # If we are also the same length, then we must be the same string (sans case)
                    hypo_seq = sequence.copy()
                    hypo_seq.append(i)

                    # Add the log prob of this token to the previous running total
                    likelihood = current_likelihood + float(log_probs[i])

                    hypo = (hypo_seq, likelihood)
                    # If we have extended to a space following the context, then that hypothesis gets to be done
                    # This takes a lot longer that just requiring extending beyond existing context
                    #last_space_pos = hypo_str.rfind(" ")
                    #if last_space_pos >= len(context):
                    # Just require hypotheses to extend beyond the existing typed context
                    if len(hypo_str) > len(context):
                        done.append(hypo)
                        # Track the most probable finishing hypothesis
                        if likelihood > done_best:
                            done_best = likelihood
                            #print(f"NEW BEST = '{hypo_str}' {hypo}")
                    else:
                        valid.append(hypo)

    done = sorted(done, key=lambda x: x[1], reverse=True)
    print(f"Search time={timer() - start_search:.2f}, done={len(done)}")

    # Print out the completed alignments from most to least probable
    print(f"Top hypotheses:")
    for i in range(min(len(done), 100)):
        hypo = done[i]
        print(f"{i:4}: {hypo[1]:8.2f} ", end="")
        combined = ""
        segmented = ""
        for j in range(1, len(hypo[0])):
            segmented = segmented + f"{hypo[0][j]} '{index_to_word[hypo[0][j]]}' "
            combined = combined + index_to_word[hypo[0][j]]
        print(f" {combined} -> {segmented}")

    # Index in the hypothesis string that is the next character after our context
    target_pos = len(context)

    start_search = timer()
    # Create a hash mapping each valid following character to a list of log probabilities
    char_to_log_probs = {}
    used = 0
    for hypo in done:
        hypo_str = ""
        # Note: Skipping index 0 since this is the space character we forced at the start
        for i in range(1, len(hypo[0])):
            hypo_str = hypo_str + index_to_word_lower[hypo[0][i]]
        #print(f"hypo_str = '{hypo_str}'")
        ch = hypo_str[target_pos]
        # Map any type of following whitespace character to be our space symbol
        if ch.isspace():
            ch = SPACE_CHAR
        # Only keep hypotheses that are something in our symbol set
        if ch in symbol_set:
            # Create an empty list if we haven't seen this character before
            if ch not in char_to_log_probs:
                char_to_log_probs[ch] = []
            char_to_log_probs[ch].append(hypo[1])
            used += 1

    # Parallel array to symbol_set for storing the marginals
    char_probs = []
    for ch in symbol_set:
        # Handle cases when symbols are never seen
        if ch in char_to_log_probs:
            char_probs.append(logsumexp(char_to_log_probs[ch]))
        else:
            char_probs.append(float("-inf"))
    # Normalize to a distribution that sums to 1
    char_probs = softmax(char_probs)

    print(f"Marginal time={timer() - start_search:.2f}, hypotheses={len(done)}, used={used}")

    width = 120
    print("             " + "_" * width)
    for i in range(len(symbol_set)):
        print(f"{symbol_set[i]} = {char_probs[i]:4.2e} " + "*" * int(char_probs[i] * width))
    print("             " + "_" * width)
