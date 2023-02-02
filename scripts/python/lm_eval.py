from bcipy.language.model.gpt2 import GPT2LanguageModel
from bcipy.language.model.kenlm import KenLMLanguageModel
from bcipy.language.model.mixture import MixtureLanguageModel
from bcipy.language.model.unigram import UnigramLanguageModel
from bcipy.language.model.causal import CausalLanguageModel
from bcipy.language.main import ResponseType
from math import log10
from timeit import default_timer as timer
from bcipy.language.main import BACKSPACE_CHAR, SPACE_CHAR, alphabet
import argparse
import numpy as np
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', dest='verbose', type=int, required=True,
        help='0: Only output model averages\n1: Output results from each phrase\n2: Output results from each character')

    parser.add_argument('--model', dest='model', type=int, required=True,
        help='0: Unigram\n1: GPT-2\n2: Mixture (80/20 GPT/Unigram)\n3: KenLM n-gram\n4: Causal Hugging Face')

    parser.add_argument('--phrases', dest='phrases', type=str, required=True,
        help='Phrase set filename')

    parser.add_argument('--model-name', dest='model_name')
    parser.add_argument("--use-mps",
                        action="store_true",
                        help="Use MPS Apple Silicon GPU during inference")
    parser.add_argument("--use-cuda",
                        action="store_true",
                        help="Use CUDA GPU during inference")
    parser.add_argument('--model-dir',
                        dest='model_dir',
                        help="Local directory to load fine-tuned causal model")

    args = parser.parse_args()

    verbose = args.verbose
    model = args.model
    phrases = args.phrases

    if model == 3 and not args.model_name:
        print(f"ERROR: For KenLM n-gram model you must specify filename of model using --model-name")
        sys.exit(1)

    if model == 4 and not args.model_name:
        print(f"ERROR: For causal model you must specify name of model using --model-name")
        sys.exit(1)

    device = "cpu"
    if args.use_mps:
        device = "mps"
    elif args.use_cuda:
        device = "cuda"

    # Read in the phrase file
    phrase_file = open(phrases, "r")
    phrases = phrase_file.readlines()
    phrase_file.close()

    lm = None

    start = timer()
    
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    if model == 0:
        lm = UnigramLanguageModel(response_type, symbol_set)
    elif model == 1:
        lm = GPT2LanguageModel(response_type, symbol_set)
    elif model == 2:
        lm = MixtureLanguageModel(response_type, symbol_set)
    elif model == 3:
        lm = KenLMLanguageModel(response_type, symbol_set, args.model_name)
    elif model == 4:
        lm = CausalLanguageModel(response_type=response_type,
                                 symbol_set=symbol_set,
                                 model_name=args.model_name,
                                 device=device,
                                 model_dir=args.model_dir)
    else:
        parser.print_help()
        exit()

    print(f"Model load time = {timer() - start:.6f}")

    phrase_count = 0
    sum_per_symbol_logprob = 0.0
    zero_prob = 0
    overall_predict_time_arr = np.array([])
    overall_predict_details_arr = np.array([])

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

        predict_time_arr = np.array([])
        predict_details_arr = np.array([])

        # Iterate over characters in phrase
        for (i, token) in enumerate(tokens):
            start_predict = timer()
            correct_char = ""

            # BciPy treats space as underscore
            if(token == "<sp>"):
                token = SPACE_CHAR
                correct_char = SPACE_CHAR
            else:
                correct_char = token.upper()
            score = 0.0    
            next_char_pred = lm.state_update(list(context))

            predict_time = timer() - start_predict
            predict_time_arr = np.append(predict_time_arr, predict_time)
            predict_details_arr = np.append(predict_details_arr, f"sentence = {sentence}, index = {i}, p( {token} | {prev_token} )")

            # Find the probability for the correct character
            p = next_char_pred[[c[0] for c in next_char_pred].index(correct_char)][1]
            if p == 0:
                zero_prob += 1
                accum = 1
                if verbose >= 2:
                    print(f"p( {token} | {prev_token} ...) = 0")
                    print(f"prediction time = {predict_time:.6f}")
                break
            else:
                score = log10(p)

                # Character-level output
                if verbose >= 2:
                    print(f"p( {token} | {prev_token} ...) = {p:.6f} [ {score:.6f} ]")
                    print(f"prediction time = {predict_time:.6f}")
                accum += score
                prev_token = token
                context += token
        
        # Compute summary stats on prediction times for this phrase
        per_symbol_time = np.average(predict_time_arr)
        phrase_std = np.std(predict_time_arr)
        phrase_max = np.max(predict_time_arr)
        phrase_min = np.min(predict_time_arr)

        # Add this phrase's prediction times to overall array
        overall_predict_time_arr = np.append(overall_predict_time_arr, predict_time_arr, axis=None)
        overall_predict_details_arr = np.append(overall_predict_details_arr, predict_details_arr, axis=None)

        if accum == 1:
            if verbose >= 1:
                print("Zero-prob event encountered, terminating phrase")
                print(f"per-symbol prediction time = {per_symbol_time:.6f} +/- {phrase_std:.6f} [{phrase_min:.6f}, {phrase_max:.6f}]\n")
        else:
            per_symbol_logprob = accum / symbols

            # Phrase-level output
            if verbose >= 1:
                print(f"sum logprob = {accum:.4f}, per-symbol logprob = {per_symbol_logprob:.4f}, ppl = {pow(10,-1 * per_symbol_logprob):.4f}")
                print(f"per-symbol prediction time = {per_symbol_time:.6f} +/- {phrase_std:.6f} [{phrase_min:.6f}, {phrase_max:.6f}]\n")
            
            sum_per_symbol_logprob += per_symbol_logprob
            phrase_count += 1

    average_per_symbol_logprob = sum_per_symbol_logprob / phrase_count

    overall_per_symbol_time = np.average(overall_predict_time_arr)
    overall_std_time = np.std(overall_predict_time_arr)
    overall_min_time = np.min(overall_predict_time_arr)
    overall_max_time = np.max(overall_predict_time_arr)

    ci_floor = overall_per_symbol_time - (3 * overall_std_time)
    ci_ceiling = overall_per_symbol_time + (3 * overall_std_time)

    # Model-level output
    print(f"OVERALL \
        \nphrases = {phrase_count}, \
        \naverage per symbol logprob = {average_per_symbol_logprob:.4f}, \
        \nppl = {pow(10,-1 * average_per_symbol_logprob):.4f}, \
        \nzero-prob events = {zero_prob} \
        \nper-symbol prediction time = {overall_per_symbol_time:.6f} +/- {overall_std_time:.6f} [{overall_min_time:.6f}, {overall_max_time:.6f}] \
        \n95% CI = [{ci_floor}, {ci_ceiling}]\n")


    for (i, time) in enumerate(overall_predict_time_arr):
        if time < ci_floor:
            print(f"LOW OUTLIER: {overall_predict_details_arr[i]}, predict time = {time:.6f}\n")
        if time > ci_ceiling:
            print(f"HIGH OUTLIER: {overall_predict_details_arr[i]}, predict time = {time:.6f}\n")