from bcipy.language.model.gpt2 import GPT2LanguageModel
from bcipy.helpers.task import alphabet
from bcipy.language.main import ResponseType
from math import log10
import sys
import kenlm

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

    # Read in the phrase file
    phrase_file = open("../lms/boston-tokenized.txt", "r")
    phrases = phrase_file.readlines()
    phrase_file.close()

    
    # Setup for the GPT2 Model
    print("\nmodel = GPT2\n")
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    lm = GPT2LanguageModel(response_type, symbol_set)

    # Stat tracking for GPT2 Model
    phrase_count = 0
    sum_per_symbol_logprob = 0.0

    # Iterate over phrases
    for phrase in phrases:
        sentence = phrase.strip()
        accum = 0.0
        
        # Phrase-level output
        if verbose >= 1:
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

            # GPT2 treats space as underscore
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

            # Character-level output
            if verbose >= 2:
                print(f"p( {token} | {prev_token} ...) = {p:.6f} [ {score:.6f} ]")
            accum += score
            prev_token = token
            context += token
        per_symbol_logprob = accum / symbols

        # Phrase-level output
        if verbose >= 1:
            print(f"sum logprob = {accum:.4f}, per-symbol logprob = {per_symbol_logprob:.4f}, ppl = {pow(10,-1 * per_symbol_logprob):.4f}\n")
        
        sum_per_symbol_logprob += per_symbol_logprob
        phrase_count += 1

    average_per_symbol_logprob = sum_per_symbol_logprob / phrase_count

    # Model-level output
    print(f"phrases = {phrase_count}, \
        average per symbol logprob = {average_per_symbol_logprob:.4f}, \
        ppl = {pow(10,-1 * average_per_symbol_logprob):.4f}\n")

    
    # Setup for the KenLM Model
    print("\nmodel = KenLM\n")
    model = kenlm.LanguageModel("../lms/lm_dec19_char_12gram_1e-5_kenlm_probing.bin")
    state = kenlm.State()
    state2 = kenlm.State()

    # Stat tracking for KenLM Model
    sum_per_symbol_logprob = 0.0
    phrase_count = 0

    # Iterate over phrases
    for phrase in phrases:
        model.BeginSentenceWrite(state)
        sentence = phrase.strip()
        accum = 0.0
        
        # Phrase-level output
        if verbose >= 1:
            print(f"sentence = '{sentence}'")

        # Split into characters
        tokens = sentence.split()
        symbols = len(tokens)

        # Initial previous token is the start symbol, initial context empty
        prev_token = "<s>"
        context = ""

        # Iterate over characters in phrase
        for i, token in enumerate(tokens):
            score = 0.0
            if i % 2 == 0:
                score = model.BaseScore(state, token, state2)
            else:
                score = model.BaseScore(state2, token, state)

            # Character-level output
            if verbose >= 2:
                print(f"p( {token} | {prev} ...) = {pow(10, score):.6f} [ {score:.6f} ]")
            accum += score
            prev = token
        
        per_symbol_logprob = accum / symbols
        
        # Phrase-level output
        if verbose >= 1:
            print(f"sum logprob = {accum:.4f}, per-symbol logprob = {per_symbol_logprob:.4f}, ppl = {pow(10,-1 * per_symbol_logprob):.4f}\n")

        sum_per_symbol_logprob += per_symbol_logprob
        phrase_count += 1

    average_per_symbol_logprob = sum_per_symbol_logprob / phrase_count

    # Model-level output
    print(f"phrases = {phrase_count}, \
        average per symbol logprob = {average_per_symbol_logprob:.4f}, \
        ppl = {pow(10,-1 * average_per_symbol_logprob):.4f}\n")