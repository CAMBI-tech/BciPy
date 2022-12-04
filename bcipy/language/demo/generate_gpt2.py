# Exploring how to use GPT-2 to generate from a given text prompt

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from timeit import default_timer as timer

if __name__ == "__main__":
    start = timer()

    # Load the GPT-2 model
    # Note: Larger models seem to return positive log probs
    lm_path = "gpt2"               #  117M parameters, 12 layers, d_model  768    ~1.3GB memory during load
    #lm_path = "gpt2-medium"        #  345M             24                 1024    ~3.1GB
    #lm_path = "gpt2-large"         #  762M             36                 1280    ~6.3GB
    #lm_path = "gpt2-xl"            # 1542M             48                 1600   ~12.3GB

    # Previous text context that we are extending
    # context = "i like zebra"  # no zebras :(
    #context = "i like zebr"   # we have zebras!
    context = "i like zeb"   # we have zebras!

    num_results = 10

    print(f"Loading {lm_path} model")
    model = GPT2LMHeadModel.from_pretrained(lm_path)
    print(f"Load time = {timer() - start:.2f}")

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()

    print(f"Loading tokenizer")
    tokenizer = GPT2TokenizerFast.from_pretrained(lm_path)
    vocab_size = tokenizer.vocab_size

    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(context, return_tensors='pt')

    print(f"{input_ids.size()}")

    # set return_num_sequences > 1
    outputs = model.generate(
        input_ids,
        max_length=input_ids.size()[1] + 1,
        num_beams=num_results,
        num_return_sequences=num_results,
        early_stopping=False,
        output_scores=True,
        return_dict_in_generate=True
    )

    for i in range(len(outputs.sequences)):
        print(f"{i:4}: {outputs.sequences_scores[i]:6.4f} {tokenizer.decode(outputs.sequences[i])} {outputs.sequences[i]}")
