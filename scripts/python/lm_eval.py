from bcipy.language.model.kenlm import KenLMLanguageModel
from bcipy.language.model.mixture import MixtureLanguageModel
from bcipy.language.model.unigram import UnigramLanguageModel
from bcipy.language.model.causal import CausalLanguageModel
from bcipy.language.model.seq2seq import Seq2SeqLanguageModel
from bcipy.language.model.classifier import ClassifierLanguageModel
from bcipy.language.main import ResponseType
from math import log10
from timeit import default_timer as timer
from bcipy.helpers.symbols import SPACE_CHAR, alphabet
import argparse
import numpy as np
import json
from scipy.stats import bootstrap
from datetime import datetime
import os

MODEL_MAPPING = {
    "unigram": UnigramLanguageModel,
    "mixture": MixtureLanguageModel,
    "kenlm": KenLMLanguageModel,
    "causal": CausalLanguageModel,
    "seq2seq": Seq2SeqLanguageModel,
    "classifier": ClassifierLanguageModel
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', dest='verbose', type=int, default=0,
        help='0: Only output model averages\n1: Output results from each phrase\n2: Output results from each character')

    model_options = MODEL_MAPPING.keys()
    parser.add_argument('--model', dest='model', required=True,
                        choices=model_options)

    parser.add_argument('--phrases', dest='phrases', type=str, required=True,
                        help='Phrase set filename')

    parser.add_argument('--model-name', dest='model_name')
    parser.add_argument('--model-dir',
                        dest='model_dir',
                        help="Local directory to load fine-tuned causal model")
    parser.add_argument("--use-mps",
                        action="store_true",
                        help="Use MPS Apple Silicon GPU during inference")
    parser.add_argument("--use-cuda",
                        action="store_true",
                        help="Use CUDA GPU during inference")
    parser.add_argument("--left-context", help="left language model context for causal model", default="")
    parser.add_argument('--add-char', help="add character to symbol set", action='append', dest="extra_chars")
    parser.add_argument("--time-outliers", help="print time outliers at end", action="store_true")
    parser.add_argument('--stats-file', help="write summary stats to specified file")
    parser.add_argument('--stats-extra', help="extra string to write to stats file as first column")
    parser.add_argument("--phrase-limit", type=int, help="max phrases to evaluate")
    parser.add_argument("--beam-width", type=int, default=8, help="search beam width")
    parser.add_argument("--batch-size", type=int, default=8, help="inference batch size")
    parser.add_argument("--token-backoff", type=int, default=-1, help="tokens removed during search, -1=last space")
    parser.add_argument('--ppl-file', help="output sentence and ppl to a file")
    parser.add_argument('--symbol-file', help="output symbol log probs to a file")
    parser.add_argument("--fp16", help="convert model to fp16 (CUDA only)", action="store_true")
    parser.add_argument("--mixed-case-context", help="use mixed case left context", action="store_true", default=False)
    parser.add_argument("--case-simple", help="simple automatic casing of let context", action="store_true", default=False)
    parser.add_argument("--ngram-lm", help="ngram model to load")
    parser.add_argument("--ngram-mix", type=float, default=0.5, help="mixture weight for ngram in type 6 mix")
    parser.add_argument('--srilm-file', help="output SRILM debug 2 log file")
    parser.add_argument("--skip-norm", help="skip normalization over symbol set for KenLM model", action="store_true", default=False)
    parser.add_argument("--mixture-params", help="JSON file path containing custom mixture model parameters", type=str)

    args = parser.parse_args()

    verbose = args.verbose
    model = args.model
    phrases = args.phrases

    # if model == "kenlm" and not args.model_dir:
    #     print("ERROR: For KenLM n-gram model you must specify filename of model using --model-dir")
    #     sys.exit(1)

    # if model == "causal" and not args.model_name:
    #     print("ERROR: For causal model you must specify name of model using --model-name")
    #     sys.exit(1)

    if args.case_simple and not args.mixed_case_context:
        print(f"WARNING: You should probably also set --mixed-case-context with --case-simple")

    # Allow passing in of space characters in the context using <sp> word
    args.left_context = args.left_context.replace("<sp>", " ")

    device = "cpu"
    if args.use_mps:
        device = "mps"
    elif args.use_cuda:
        device = "cuda"

    # Read in the phrase file
    phrase_file = open(phrases, "r")
    phrases = phrase_file.readlines()
    phrase_file.close()

    # We may want to limit to only the first so many phrases
    if args.phrase_limit:
        while len(phrases) > args.phrase_limit:
            phrases.pop()

    lm = None

    ppl_file = None
    if args.ppl_file:
        ppl_file = open(args.ppl_file, "w")

    # Optional output of a log file in the same format as SRILM at debug level 2.
    # This allows us to compute a mixture weight based on the multiple log files using compute-best-mix script.
    srilm_file = None
    if args.srilm_file:
        srilm_file = open(args.srilm_file, "w")

    start = timer()

    symbol_set = alphabet()
    if args.extra_chars:
        for char in args.extra_chars:
            symbol_set += char
        print(f"Modified symbol_set: {symbol_set}")

    response_type = ResponseType.SYMBOL

    kwargs = {
        "lm_path": args.model_dir,
        "skip_symbol_norm": args.skip_norm,
        "lang_model_name": args.model_name,
        "lm_device": device,
        "lm_left_context": args.left_context,
        "beam_width": args.beam_width,
        "batch_size": args.batch_size,
        "token_backoff": args.token_backoff,
        "fp16": args.fp16,
        "mixed_case_context": args.mixed_case_context,
        "case_simple": args.case_simple,
    }

    mixture_params = dict()
    if args.mixture_params is not None:
        with open(args.mixture_params, 'r') as file:
            mixture_params = json.load(file)


    lm = MODEL_MAPPING[model](response_type, symbol_set, **kwargs, **mixture_params)

    # elif model == 6:
    #     lm = MixtureLanguageModel(response_type=response_type,
    #                              symbol_set=symbol_set,
    #                              lm_types=["CAUSAL", "KENLM"],
    #                              lm_weights=[1.0 - args.ngram_mix, args.ngram_mix],
    #                              lm_params=[{"lang_model_name": args.model_name,
    #                                          "lm_device": device,
    #                                          "lm_path": args.model_dir,
    #                                          "lm_left_context": args.left_context,
    #                                          "beam_width": args.beam_width,
    #                                          "batch_size": args.batch_size,
    #                                          "token_backoff": args.token_backoff,
    #                                          "fp16": args.fp16,
    #                                          "mixed_case_context": args.mixed_case_context,
    #                                          "case_simple": args.case_simple
    #                                         },
    #                                         {"lm_path": args.ngram_lm}])
    # else:
    #     parser.print_help()
    #     exit()

    print(f"Model load time = {timer() - start:.2f}")

    phrase_count = 0
    sum_per_symbol_logprob = 0.0
    zero_prob = 0
    overall_predict_time_arr = np.array([])
    overall_predict_details_arr = np.array([])

    start = timer()

    sum_log_prob = 0.0
    sum_symbols = 0
    all_symbol_log_probs = []

    # Iterate over phrases
    for phrase in phrases:
        sentence = phrase.strip()
        if len(sentence) > 0:
            accum = 0.0

            # Phrase-level output
            if verbose >= 1:
                print(f"sentence = '{sentence}'")

            # Split into characters
            tokens = sentence.split()
            symbols = len(tokens)

            # SRILM starts with the sentence being evaluated
            if srilm_file:
                for (i, symbol) in enumerate(tokens):
                    if i > 0:
                        srilm_file.write(" ")
                    srilm_file.write(symbol.replace("_", "<sp>"))
                srilm_file.write("\n")

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
                if (token == "<sp>"):
                    token = SPACE_CHAR
                    correct_char = SPACE_CHAR
                else:
                    correct_char = token.upper()
                score = 0.0
                next_char_pred = lm.state_update(list(context))

                predict_time = timer() - start_predict
                predict_time_arr = np.append(predict_time_arr, predict_time)
                predict_details_arr = np.append(predict_details_arr,
                                                f"sentence = {sentence}, index = {i}, p( {token} | {prev_token} )")

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

                # SRILM line for a character looks like: "	p( w | <s> ) 	= [2gram] 0.095760 [ -1.018816 ]"
                if srilm_file:
                    extra = ""
                    if i > 0:
                        extra = " ..."
                    # The 1gram bit is only relevant for the n-gram, we'll just hard code to 1gram for everything
                    srilm_file.write(f"\tp( {token.replace('_', '<sp>')} | {prev_token.replace('_', '<sp>')}{extra}) \t= [1gram] {p:.6f} [ {score:.6f} ]\n")

                accum += score
                prev_token = token
                context += token
                all_symbol_log_probs.append(score)

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
                    print(f"per-symbol prediction time = {per_symbol_time:.6f} +/- {phrase_std:.6f} [{phrase_min:.6f}, "
                          f"{phrase_max:.6f}]\n")
            else:
                per_symbol_logprob = accum / symbols
                sent_ppl = pow(10, -1 * per_symbol_logprob)

                # Phrase-level output
                if verbose >= 1:
                    print(f"sum logprob = {accum:.4f}, per-symbol logprob = {per_symbol_logprob:.4f}, ppl = {sent_ppl:.4f}")
                    print(f"per-symbol prediction time = {per_symbol_time:.6f} +/- {phrase_std:.6f} [{phrase_min:.6f}, {phrase_max:.6f}]\n")

                sum_per_symbol_logprob += per_symbol_logprob
                phrase_count += 1

                # Optional output to a file with a sentence and its ppl and log prob
                if ppl_file:
                    ppl_file.write(f"{sent_ppl:.4f}\t{accum:.4f}\t{sentence}\n")
                    ppl_file.flush()

                # To calculate the overall file perplexity, we need the sum of log probs of all sentences.
                # This is how SRILM does it and makes it less sensitive to particular outlier sentences.
                sum_log_prob += accum
                sum_symbols += symbols

        # SRILM state for the sentence
        if srilm_file:
            srilm_file.write(f"0 sentences, {symbols} words, 0 OOVs\n")
            srilm_file.write(f"0 zeroprobs, logprob= {accum:.4f} ppl= {sent_ppl:.3f} ppl1= {sent_ppl:.3f}\n")
            srilm_file.write("\n")
            srilm_file.flush()

    inference_time = timer() - start

    if ppl_file:
        ppl_file.close()

    overall_per_symbol_time = np.average(overall_predict_time_arr)
    overall_std_time = np.std(overall_predict_time_arr)
    overall_min_time = np.min(overall_predict_time_arr)
    overall_max_time = np.max(overall_predict_time_arr)

    ci_floor = overall_per_symbol_time - (2 * overall_std_time)
    ci_ceiling = overall_per_symbol_time + (2 * overall_std_time)

    ppl = float("+inf")
    if sum_symbols > 0:
        ppl = pow(10, -1 * sum_log_prob / sum_symbols)

    # SRILM final overall stats lines
    if srilm_file:
        srilm_file.write(f"file {args.phrases}: 0 sentences, {sum_symbols} words, 0 OOVs\n")
        srilm_file.write(f"0 zeroprobs, logprob= {sum_log_prob:.4f} ppl= {ppl:.3f} ppl1= {ppl:.3f}\n")
        srilm_file.close()

    # Model-level output
    print(f"OVERALL \
        \nphrases = {phrase_count}, \
        \nzero-prob events = {zero_prob} \
        \nper-symbol prediction time = {overall_per_symbol_time:.6f} +/- {overall_std_time:.6f} [{overall_min_time:.6f}, {overall_max_time:.6f}] \
        \n95% CI = [{ci_floor:.6f}, {ci_ceiling:.6f}] \
        \ninference time = {inference_time:.2f}\
        \nsum logprob = {sum_log_prob:.2f} \
        \nsum symbols = {sum_symbols} \
        \nmean symbol log prob = {np.average(all_symbol_log_probs):.4f} \
        \nppl = {ppl:.4f}")

    # Optional fill that contains the log prob of each prediction
    # Could be useful for recomputing confidence intervals or such
    if args.symbol_file:
        symbol_file = open(args.symbol_file, "w")
        for log_prob in all_symbol_log_probs:
            symbol_file.write(str(log_prob) + "\n")
        symbol_file.close()

    if args.stats_file:
        # Single line file output, useful for running experiments
        print(f"Outputting stats to {args.stats_file}, running bootstrap on {len(all_symbol_log_probs)} samples.")
        time_bootstrap = timer()
        bootstrap_log_prob = bootstrap(data=(all_symbol_log_probs,),
                                        statistic=np.mean,
                                        confidence_level=0.95)
        print(f"Bootstrap completed in {(timer() - time_bootstrap):.2f} seconds.")

        ppl_high = pow(10, -1 * bootstrap_log_prob.confidence_interval.low)
        ppl_low = pow(10, -1 * bootstrap_log_prob.confidence_interval.high)
        error_bar = (ppl_high - ppl_low) / 2.0

        extra = ""
        extra_col = ""
        if args.stats_extra:
            extra = args.stats_extra + "\t"
            extra_col = "\t"
        params = -1
        if model == 4:
            params = lm.get_num_parameters()

        exists = os.path.isfile(args.stats_file)
        with open(args.stats_file, 'a') as file:
            if not exists:
                # Header if the stats file doesn't already exist
                file.write(f"{extra_col}ppl\tsum_log_prob\tsum_symbols\tboot_ppl_pm\tboot_ppl_low\tboot_ppl_high\tphrases\ttime\tparams\tdate_time\n")
            file.write(f"{extra}"
                         f"{ppl:.6f}"
                         f"\t{sum_log_prob:.6f}"
                         f"\t{sum_symbols}"
                         f"\t{error_bar:.6f}"
                         f"\t{ppl_low:.6f}"
                         f"\t{ppl_high:.6f}"
                         f"\t{phrase_count}"
                         f"\t{inference_time:.6f}"
                         f"\t{params}"
                         f"\t{datetime.now()}\n")

    # Optionally print the predictions that took an abnormal amount of time
    if args.time_outliers:
        for (i, time) in enumerate(overall_predict_time_arr):
            if time < ci_floor:
                print(f"LOW OUTLIER: {overall_predict_details_arr[i]}, predict time = {time:.6f}\n")
            if time > ci_ceiling:
                print(f"HIGH OUTLIER: {overall_predict_details_arr[i]}, predict time = {time:.6f}\n")
