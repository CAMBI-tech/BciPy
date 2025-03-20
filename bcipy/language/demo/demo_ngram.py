# Basic sanity test of using KenLM to predict a sentence using a 12-gram character model.

from bcipy.language.model.ngram import NGramLanguageModelAdapter
from bcipy.core.symbols import DEFAULT_SYMBOL_SET
from bcipy.language.main import ResponseType
from bcipy.config import LM_PATH
from bcipy.exceptions import KenLMInstallationException

try:
    import kenlm
except BaseException:
    raise KenLMInstallationException(
        "Please install the requisite kenlm package:\n'pip install kenlm==0.1 --global-option=\"--max_order=12\"")


if __name__ == "__main__":
    lm_path = f"{LM_PATH}/lm_dec19_char_12gram_1e-5_kenlm_probing.bin"

    # Using KenLM directly

    # Load a really pruned n-gram language model
    model = kenlm.LanguageModel(lm_path)

    # Sum of the log prob of the sentence: <s> i like zebras. </s>
    # Results using SRILM ngram utility:
    # % ngram -lm lm_dec19_char_12gram_1e-5.arpa -order 12 -ppl zebras.txt -debug 2
    # reading 34 1-grams
    # reading 661 2-grams
    # reading 4149 3-grams
    # reading 4609 4-grams
    # reading 5873 5-grams
    # reading 6170 6-grams
    # reading 7768 7-grams
    # reading 7748 8-grams
    # reading 8202 9-grams
    # reading 7702 10-grams
    # reading 7953 11-grams
    # reading 8026 12-grams
    # <s> i <sp> l i k e <sp> z e b r a s . </s>
    # 	p( i | <s> ) 	= [2gram] 0.234358 [ -0.630121 ]
    # 	p( <sp> | i ...) 	= [3gram] 0.503908 [ -0.297649 ]
    # 	p( l | <sp> ...) 	= [4gram] 0.040375 [ -1.393890 ]
    # 	p( i | l ...) 	= [5gram] 0.441208 [ -0.355357 ]
    # 	p( k | i ...) 	= [6gram] 0.896195 [ -0.047597 ]
    # 	p( e | k ...) 	= [7gram] 0.999651 [ -0.000152 ]
    # 	p( <sp> | e ...) 	= [8gram] 0.929917 [ -0.031556 ]
    # 	p( z | <sp> ...) 	= [2gram] 0.000114 [ -3.941586 ]
    # 	p( e | z ...) 	= [2gram] 0.410016 [ -0.387199 ]
    # 	p( b | e ...) 	= [2gram] 0.002148 [ -2.668019 ]
    # 	p( r | b ...) 	= [2gram] 0.044926 [ -1.347505 ]
    # 	p( a | r ...) 	= [2gram] 0.043585 [ -1.360661 ]
    # 	p( s | a ...) 	= [2gram] 0.070985 [ -1.148835 ]
    # 	p( . | s ...) 	= [2gram] 0.032252 [ -1.491447 ]
    # 	p( </s> | . ...) 	= [3gram] 0.728636 [ -0.137489 ]
    # 1 sentences, 14 words, 0 OOVs
    # 0 zeroprobs, logprob= -15.2391 ppl= 10.374 ppl1= 12.260
    #
    # file zebras.txt: 1 sentences, 14 words, 0 OOVs
    # 0 zeroprobs, logprob= -15.2391 ppl= 10.374 ppl1= 12.260
    sentence = "i <sp> l i k e <sp> z e b r a s ."
    print(f"Sentence '{sentence}', logprob = {model.score(sentence, bos=True, eos=True):.4f}\n")

    # Stateful query going one token at-a-time
    # We'll flip flop between two state objects, one is the input and the other is the output
    state = kenlm.State()
    state2 = kenlm.State()

    model.BeginSentenceWrite(state)
    accum = 0.0

    tokens = sentence.split()
    tokens.append("</s>")
    prev = "<s>"
    for i, token in enumerate(tokens):
        score = 0.0
        if i % 2 == 0:
            score = model.BaseScore(state, token, state2)
        else:
            score = model.BaseScore(state2, token, state)
        print(f"p( {token} | {prev} ...) = {pow(10, score):.6f} [ {score:.6f} ]")
        accum += score
        prev = token
    print(f"sum logprob = {accum:.4f}")

    # Using the adapter and aactextpredict toolkit
    response_type = ResponseType.SYMBOL
    lm = NGramLanguageModelAdapter(response_type, DEFAULT_SYMBOL_SET, lm_path)

    print("Target sentence: i_like_zebras\n")

    next_char_pred = lm.predict(list("i_like_z"))
    print("Context: i_like_z")
    print(f"Predictions: {next_char_pred}")
    correct_char_rank = [c[0] for c in next_char_pred].index("E") + 1
    print(f"Correct character rank: {correct_char_rank}\n")

    next_char_pred = lm.predict(list("i_lik"))
    print("Context: i_lik")
    print(f"Predictions: {next_char_pred}")
    correct_char_rank = [c[0] for c in next_char_pred].index("E") + 1
    print(f"Correct character rank: {correct_char_rank}\n")

    next_char_pred = lm.predict(list("i_like_zebras"))
    print("Context: i_like_zebras")
    print(f"Predictions: {next_char_pred}")
    correct_char_rank = [c[0] for c in next_char_pred].index("_") + 1
    print(f"Correct character rank: {correct_char_rank}\n")

    next_char_pred = lm.predict(list(""))
    print("Context: ")
    print(f"Predictions: {next_char_pred}")
    correct_char_rank = [c[0] for c in next_char_pred].index("I") + 1
    print(f"Correct character rank: {correct_char_rank}\n")

    next_char_pred = lm.predict(list("i_like_zebra"))
    print("Context: i_like_zebra")
    print(f"Predictions: {next_char_pred}")
    correct_char_rank = [c[0] for c in next_char_pred].index("S") + 1
    print(f"Correct character rank: {correct_char_rank}\n")
