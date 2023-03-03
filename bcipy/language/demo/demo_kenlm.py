# Basic sanity test of using KenLM to predict a sentence using a 12-gram character model.

import kenlm
import os
from bcipy.language.model.kenlm import KenLMLanguageModel
from bcipy.language.main import alphabet
from bcipy.language.main import ResponseType

if __name__ == "__main__":
    dirname = os.path.dirname(__file__) or '.'
    lm_path = f"{dirname}/../lms/lm_dec19_char_12gram_1e-5_kenlm_probing.bin"

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

    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    lm = KenLMLanguageModel(response_type, symbol_set, lm_path)

    next_char_pred = lm.state_update(list("i_like_z"))
    print(next_char_pred)
    correct_char_rank = [c[0] for c in next_char_pred].index("E") + 1
    print(correct_char_rank)
    next_char_pred = lm.state_update(list("i_lik"))
    print(next_char_pred)
    correct_char_rank = [c[0] for c in next_char_pred].index("E") + 1
    print(correct_char_rank)
    next_char_pred = lm.state_update(list("i_like_zebras"))
    print(next_char_pred)
    correct_char_rank = [c[0] for c in next_char_pred].index("_") + 1
    print(correct_char_rank)
    next_char_pred = lm.state_update(list(""))
    print(next_char_pred)
    correct_char_rank = [c[0] for c in next_char_pred].index("I") + 1
    print(correct_char_rank)
    next_char_pred = lm.state_update(list("i_like_zebra"))
    print(next_char_pred)
    correct_char_rank = [c[0] for c in next_char_pred].index("S") + 1
    print(correct_char_rank)
