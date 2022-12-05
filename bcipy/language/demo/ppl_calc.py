from bcipy.language.model.gpt2 import GPT2LanguageModel
from bcipy.language.model.kenlm import KenLMLanguageModel
from bcipy.language.model.mixture import MixtureLanguageModel
from bcipy.language.model.unigram import UnigramLanguageModel
from bcipy.helpers.task import alphabet
from bcipy.language.main import ResponseType
from math import log10
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', dest='verbose', type=int, required=True,
        help='0: Only output model averages\n1: Output results from each phrase\n2: Output results from each character')

    parser.add_argument('--model', dest='model', type=int, required=True,
        help='0: Unigram\n1: GPT-2\n2: Mixture (80/20 GPT/Unigram)\n3: KenLM Tiny\n4: KenLM Large')

    parser.add_argument('--phrases', dest='phrases', type=str, required=True,
        help='Phrase set filename')

    args = parser.parse_args()
    verbose = args.verbose
    model = args.model
    phrases = args.phrases

    # Read in the phrase file
    phrase_file = open(phrases, "r")
    phrases = phrase_file.readlines()
    phrase_file.close()

    lm = None
    
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    if model == 0:
        lm = UnigramLanguageModel(response_type, symbol_set)
    elif model == 1:
        lm = GPT2LanguageModel(response_type, symbol_set)
    elif model == 2:
        lm = MixtureLanguageModel(response_type, symbol_set)
    elif model == 3:
        lm = KenLMLanguageModel(response_type, symbol_set)
    elif model == 4:
        lm - KenLMLanguageModel(response_type, symbol_set, '../lms/lm_dec19_char_large_12gram.arpa')
    else:
        parser.print_help()
        exit()

    phrase_count = 0
    sum_per_symbol_logprob = 0.0
    zero_prob = 0

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

            # BciPy treats space as underscore
            if(token == "<sp>"):
                token = "_"
                correct_char = "_"
            else:
                correct_char = token.upper()
            score = 0.0    
            next_char_pred = lm.state_update(list(context))

            # Find the probability for the correct character
            p = next_char_pred[[c[0] for c in next_char_pred].index(correct_char)][1]
            if p == 0:
                zero_prob += 1
                accum = 1
                if verbose >= 2:
                    print(f"p( {token} | {prev_token} ...) = 0")
                break
            else:
                score = log10(p)

                # Character-level output
                if verbose >= 2:
                    print(f"p( {token} | {prev_token} ...) = {p:.6f} [ {score:.6f} ]")
                accum += score
                prev_token = token
                context += token
        
        if accum == 1:
            if verbose >= 1:
                print("Zero-prob event encountered, terminating phrase")
        else:
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
        ppl = {pow(10,-1 * average_per_symbol_logprob):.4f}, \
        zero-prob events = {zero_prob}\n")
