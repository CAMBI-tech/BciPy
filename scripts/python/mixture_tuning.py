import argparse
from itertools import permutations
import numpy as np
from math import log10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', dest='step', type=int, required=True,
        help='The number of weight steps to be used in the sweep search')

    parser.add_argument('--in', dest='files', type=str, required=True, nargs='+',
        help="Two or more paths pointing to output files from the lm_eval script with --verbose 2")

    args = parser.parse_args()

    # weights = []
    # for i in range(len(args.files)):
    #     weights.append([])

    step = 1.0 / args.step
    possible_weights = np.round(list(np.arange(0, 1.0 + step, step)), 5)
    perms = list(permutations(possible_weights, len(args.files)))

    weights = [perm for perm in perms if (np.sum(perm) == 1)]

    files = []
    for f in args.files:
        files.append(open(f, 'r'))

    lines = []
    for f in files:
        lines.append(f.readlines())
        f.close()

    probs = []
    sentence_pslp_arr = []
    for line in zip(*lines):
        if line[0].startswith("sentence = "):
            for l in line:
                if not l.startswith("sentence = "):
                    print("Line mismatch. Please ensure all files were run on the same phrase set.")
                    exit()
            probs = []
        elif line[0].startswith("p( "):
            for l in line:
                if not l.startswith("p( "):
                    print("Line mismatch. Please ensure all files were run on the same phrase set.")
                    exit()
                if l[-5:] == " = 0\n":
                    print("Error: Component model zero prob.")
                    exit()
            component_probs = []
            mixed_probs = []
            for l in line:
                component_probs.append(float(l[l.index("=") + 2:l.index("[") - 1]))
            for w in weights:
                mixed_probs.append(np.dot(w, component_probs))
            probs.append(mixed_probs)
        elif line[0].startswith("sum logprob = "):
            for l in line:
                if not l.startswith("sum logprob = "):
                    print("Line mismatch. Please ensure all files were run on the same phrase set.")
                    exit()
            logprobs = []
            for p in probs:
                l = [log10(i) for i in p]
                logprobs.append(l)
            sentence_pslp_arr.append([np.average(w) for w in zip(*logprobs)])
        elif line[0].startswith("OVERALL"):
            for l in line:
                if not l.startswith("OVERALL"):
                    print("Line mismatch. Please ensure all files were run on the same phrase set.")
                    exit()

            overall_ppl_arr = [pow(10, -1 * np.average(w)) for w in zip(*sentence_pslp_arr)]
            print("Weight Distribution\tAverage Char Perplexity")
            for weight, ppl in zip(weights, overall_ppl_arr):
                print(f"{weight}\t\t{ppl:.3f}")

            print()
            print(f"BEST: {weights[np.argmin(overall_ppl_arr)]} {np.min(overall_ppl_arr):.3f}")
            break
        else:
            continue
