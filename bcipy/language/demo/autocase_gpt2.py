# Exploring how to use GPT-2 to autocase a lowercase string.
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from timeit import default_timer as timer

if __name__ == "__main__":
    start = timer()

    # Load the GPT-2 model
    # Note: Larger models seem to return positive log probs
    lm_path = "gpt2"               #  117M parameters, 12 layers, d_model  768    ~1.3GB memory during load
    #lm_path = "gpt2-medium"        #  345M             24                 1024    ~3.1GB
    #lm_path = "gpt2-large"         #  762M             36                 1280    ~6.3GB
    #lm_path = "gpt2-xl"            # 1542M             48                 1600   ~12.3GB

    beam_width = 16

    # Encode the text into its subword tokens
    # Note: There seem to be multiple subword tokens for things like . ? !
    #text = "i like my iphone"
    #text = "on monday bob and i bought iphones in the usa"
    #text = "on monday bob and i bought iphones"
    #text = "the nasa shuttle crashed"
    #text = "i love the nasa-tlx test"
    text = "i like a tale of two cities by charles dickens"

    print(f"Loading {lm_path} model")
    model = GPT2LMHeadModel.from_pretrained(lm_path)
    print(f"Load time = {timer() - start:.2f}")

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()

    print(f"Loading tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained(lm_path)
    vocab_size = tokenizer.vocab_size

    # Hash that maps numeric subword token indexes to the actual text
    # Also save a lowercase version for use in case insensitve comparisons
    print(f"Creating index to vocab map, size = {vocab_size}")
    index_to_word = {}
    index_to_word_lower = {}
    for i in range(vocab_size):
        word = tokenizer.decode([i])
        index_to_word[i] = word
        index_to_word_lower[i] = word.lower()

    # Get the index we use for the start or end pseudo-word
    start_end_index = tokenizer.encode("<|endoftext|>")[0]

    # Index of the space character
    space_index = tokenizer.encode(" ")[0]

    # If you have a GPU, put everything on cuda
    device = "cpu"
    # device = "cuda"   # NVidia GPU
    # device = "mps"    # M1 mac
    model.to(device)

    start_search = timer()

    # List of subword sequences that can produce the text
    done = []

    """This version searches for all possible segmentations, expensive!
    # List of current valid subword sequences that haven't yet completed the sentence
    current = [([start_end_index], 0.0)]

    # Keep going until we are out of viable sequences
    while len(current) > 0:
        # Get the new sequence to work on
        sequence = current.pop()[0]
        #print(f"**** sequence = {sequence}")
        tokens_tensor = torch.tensor(sequence).to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        # Create sequence text before token, skipping start word
        sequence_text = ""
        for i in range(1, len(sequence)):
            sequence_text = sequence_text + index_to_word_lower[sequence[i]]
        #print(f"sequence_text = '{sequence_text}'")

        # Create a list of token indexes that are a prefix of target text
        valid = []
        for i in range(predictions.size()[1]):
            hypo_str = sequence_text + index_to_word_lower[i]
            if text.startswith(hypo_str):
                #print(f"hypo_str = '{hypo_str}', {i}: '{index_to_word[i]}' {predictions[-1, i]:.4f}")
                # If we are also the same length, then we must be the same string (sans case)
                hypo_seq = sequence.copy()
                hypo_seq.append(i)
                hypo = (hypo_seq, float(predictions[-1, i]))
                if len(text) == len(hypo_str):
                    done.append(hypo)
                else:
                    valid.append(hypo)
#        print(f"VALID {valid}")
#        print(f"DONE {done}")
        current.extend(valid)
#        print(f"CURRENT {current}")
    """

    print(f"Starting search, beam = {beam_width}")
    # Seems more stable starting with a space than endoftext token
    valid = [([space_index], 0.0)]
    while len(valid) > 0:
        # Only work on the top hypotheses from the last round of extension
        current = sorted(valid, key=lambda x: x[1], reverse=True)
        before = len(current)
        while len(current) > beam_width:
            current.pop(-1)
        print(f"current, before={before}, after= {len(current)}, first={current[0]}, last={current[-1]}")

        # Add new extended hypotheses to this list
        valid = []

        # Keep going until we have extended all hypotheses in the current set
        while len(current) > 0:
            # Get the new sequence to work on
            (sequence, current_likelihood) = current.pop()
            tokens_tensor = torch.tensor(sequence).unsqueeze(0).to(device)
            #print(f"tokens_tensor = {tokens_tensor.size()}")
            with torch.no_grad():
                logits = model(tokens_tensor).logits
                log_probs = torch.log(torch.softmax(logits[:, -1, :].flatten(), dim=0))
                #sum_logits = torch.sum(predictions, dim=0)
                #print(f"predictions = {logits.size()}")

            # Create sequence text before token, skipping start word, make it all lowercase
            sequence_text = ""
            for i in range(1, len(sequence)):
                sequence_text = sequence_text + index_to_word_lower[sequence[i]]

            # Create a list of token indexes that are a prefix of target text
            for i in range(logits.size()[2]):
                hypo_str = sequence_text + index_to_word_lower[i]
                if text.startswith(hypo_str):
                    #print(f"hypo_str = '{hypo_str}', {i}: '{index_to_word[i]}' {predictions[-1, i]:.4f}")
                    # If we are also the same length, then we must be the same string (sans case)
                    hypo_seq = sequence.copy()
                    hypo_seq.append(i)
                    #hypo = (hypo_seq, float(logits[-1, i]))
                    #hypo = (hypo_seq, sum_logits[i])
                    likelihood = current_likelihood + log_probs[i]
                    hypo = (hypo_seq, likelihood)

                    # If we reach the length of the target text then this sequence has finished
                    if len(text) == len(hypo_str):
                        done.append(hypo)
                        #print(f"DONE, '{hypo_str}', ", end="")
                        #for j in range(0, logits.size()[1]):
                        #    print(f"{logits[0][j][i]} ", end="")
                        #print()
                        #print(f"{i:4}: {hypo[1]:8.2f} ", end="")
                        #combined = ""
                        #segmented = ""
                        #for j in range(0, len(hypo[0])):
                        #    segmented = segmented + f"{hypo[0][j]} '{index_to_word[hypo[0][j]]}' "
                        #    combined = combined + index_to_word[hypo[0][j]]
                        #print(f" {combined} -> {segmented}")

                    else:
                        valid.append(hypo)

    done = sorted(done, key=lambda x: x[1], reverse=True)
    print(f"Search time = {timer() - start_search:.2f}")

    # Print out the completed alignments from most to least probable
    for i, hypo in enumerate(done):
        print(f"{i:4}: {hypo[1]:8.2f} ", end="")
        combined = ""
        segmented = ""
        for j in range(1, len(hypo[0])):
            segmented = segmented + f"{hypo[0][j]} '{index_to_word[hypo[0][j]]}' "
            combined = combined + index_to_word[hypo[0][j]]
        print(f" {combined} -> {segmented}")

