# Exploring how to use ByT5 to generate missing spans

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from timeit import default_timer as timer

if __name__ == "__main__":

    # google/byt5-small
    # google/byt5-base
    # google/byt5-large     9GB
    # google/byt5-xl
    # google/byt5-xxl

    lm_path = "google/byt5-large"
    num_results = 20

    start = timer()
    print(f"Loading {lm_path} model, ", end="")
    model = AutoModelForSeq2SeqLM.from_pretrained(lm_path)
    print(f"time {timer() - start:.2f}")

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    # Does this matter for ByT5?
    model.eval()

    start = timer()
    print("Loading tokenizer, ", end="")
    tokenizer = AutoTokenizer.from_pretrained(lm_path)
    print(f"time {timer() - start:.2f}")

    vocab_size = tokenizer.vocab_size

    # Hash that maps numeric subword token indexes to the actual text
    # Also save a lowercase version for use in case insensitive comparisons
    print(f"Creating index to vocab map, size = {vocab_size}")
    index_to_word = {}
    index_to_word_lower = {}
    for i in range(vocab_size):
        word = tokenizer.decode([i])
        index_to_word[i] = word
        # print(f"{i:6}: '{word}'")
        index_to_word_lower[i] = word.lower()

    # encode context the generation is conditioned on
    context = " I "
    input_ids = tokenizer(context).input_ids
    print(f"input_ids, size {len(input_ids)}, {input_ids}")

    # After span token, add a space
#    input_ids_with_span = torch.tensor([input_ids[0:-1] + [258] + [35]])

    # After span token, add a space and </s>
    input_ids_with_span = torch.tensor([input_ids[0:-1] + [258] + [35] + [1]])

    # Just with the span token at the very end
#    input_ids_with_span = torch.tensor([input_ids[0:-1] + [258]])

    print(f"input_ids_with_span, size {len(input_ids_with_span)}, {input_ids_with_span}")

    start = timer()
    outputs = model.generate(
        input_ids_with_span,
        max_length=3,
        num_beams=num_results,
        num_return_sequences=num_results,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True
    )
    print(f"Generate, beam {num_results}, time {timer() - start:.2f}")

    for i in range(len(outputs.sequences)):
        print(f"{i:4}: {outputs.sequences_scores[i]:6.4f} {tokenizer.decode(outputs.sequences[i])} \
            {outputs.sequences[i]}")
