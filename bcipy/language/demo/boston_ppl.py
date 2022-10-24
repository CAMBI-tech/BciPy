from bcipy.language.model.gpt2 import GPT2LanguageModel
from bcipy.helpers.task import alphabet
from bcipy.language.main import ResponseType
from math import log10
import sys

if __name__ == "__main__":

    # How much do we output?
    verbose = 0
    if(len(sys.argv) > 1 and sys.argv[1] == "--verbose"):
        if(sys.argv[2] == "0"):
            verbose = 0
        elif (sys.argv[2] == "1"):
            verbose = 1
        elif (sys.argv[2] == "2"):
            verbose = 2
        else:
            print("\nUse '--verbose <level>' runtime argument for output control")
            print("'--verbose 0'\tOnly output model averages")
            print("'--verbose 1'\tOutput results from each phrase")
            print("'--verbose 2'\tOutput results from each character")
    else:
        print("\nUse '--verbose <level>' runtime argument for output control")
        print("'--verbose 0'\tOnly output model averages")
        print("'--verbose 1'\tOutput results from each phrase")
        print("'--verbose 2'\tOutput results from each character")

    print("\nmodel = GPT2\n")
    
    # Setup for the GPT2 Model
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    lm = GPT2LanguageModel(response_type, symbol_set)

    # Read in the phrase file
    phrase_file = open("../lms/boston-tokenized.txt", "r")
    phrases = phrase_file.readlines()
    phrase_file.close()

    phrase_count = 0
    sum_per_symbol_logprob = 0.0

    # Iterate over phrases
    for phrase in phrases:
        sentence = phrase.strip()
        accum = 0.0

        print(f"sentence = '{sentence}'")

        # Split into characters
        tokens = sentence.split()
        symbols = len(tokens)

        # Initial previous token is the start symbol, initial context empty
        prev_token = "<s>"
        context = ""

        # Iterate over characters in phrase
        for token in tokens:
            correct_char = ""
            if(token == "<sp>"):
                token = "_"
                correct_char = "_"
            else:
                correct_char = token.upper()
            score = 0.0    
            next_char_pred = lm.state_update(list(context))

            # Find the probability for the correct character
            p = next_char_pred[[c[0] for c in next_char_pred].index(correct_char)][1]
            score = log10(p)
            if verbose == 2:
                print(f"p( {token} | {prev_token} ...) = {p:.6f} [ {score:.6f} ]")
            accum += score
            prev_token = token
            context += token
        per_symbol_logprob = accum / symbols
        if verbose >= 1:
            print(f"sum logprob = {accum:.4f}, per-symbol logprob = {per_symbol_logprob:.4f}, ppl = {pow(10,-1 * per_symbol_logprob):.4f}\n")
        
        sum_per_symbol_logprob += per_symbol_logprob
        phrase_count += 1

    average_per_symbol_logprob = sum_per_symbol_logprob / phrase_count
    print(f"phrases = {phrase_count}, \
        average per symbol logprob = {average_per_symbol_logprob:.4f}, \
        ppl = {pow(10,-1 * average_per_symbol_logprob):.4f}\n")