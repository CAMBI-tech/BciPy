# Exploring how to use T5 switch transformers

from transformers import T5Tokenizer, SwitchTransformersForConditionalGeneration

from timeit import default_timer as timer

if __name__ == "__main__":

    lm_path = "google/switch-base-8"
    num_results = 100

    start = timer()
    print(f"Loading {lm_path} model, ", end="")
    model = SwitchTransformersForConditionalGeneration.from_pretrained(lm_path)
    print(f"time {timer() - start:.2f}")

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    # Does this matter for T5?
    model.eval()

    start = timer()
    print(f"Loading tokenizer, ", end="")
    tokenizer = T5Tokenizer.from_pretrained(lm_path)
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
        #print(f"{i:6}: '{word}'")
        index_to_word_lower[i] = word.lower()

    # encode context the generation is conditioned on
    #context = "The dog was walking in the "
    context = "Why "
    input_ids = tokenizer(context + "<extra_id_0>", return_tensors="pt").input_ids
    print(f"input_ids, size {input_ids.size()}, {input_ids}")

    start = timer()
    outputs = model.generate(
        input_ids,
        max_length=3,                       # <pad> <extra_id_0> word (but this won't handle if the current word is long)
        num_beams=num_results,
        num_return_sequences=num_results,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True
    )
    print(f"Generate, beam {num_results}, time {timer() - start:.2f}")

    for i in range(len(outputs.sequences)):
        print(f"{i:4}: {outputs.sequences_scores[i]:6.4f} {tokenizer.decode(outputs.sequences[i])} {outputs.sequences[i]}")
