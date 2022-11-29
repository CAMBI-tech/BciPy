from bcipy.language.model.mixture import MixtureLanguageModel
from bcipy.helpers.task import alphabet
from bcipy.language.main import ResponseType
from math import log10
from bcipy.language.uniform import (ResponseType, UniformLanguageModel,
                                    equally_probable)

if __name__ == "__main__":

    print("\nmodel = GPT2 / Unigram Mixture\n")
    
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    lm = MixtureLanguageModel(response_type, symbol_set)

    
    sentence = "i <sp> l i k e <sp> z e b r a s"
    accum = 0.0

    print(f"sentence = '{sentence}'")

    tokens = sentence.split()
    symbols = len(tokens)
    prev = "<s>"
    context = ""


    for i,token in enumerate(tokens):
        correct_char = ""
        if(token == "<sp>"):
            token = "_"
            correct_char = "_"
        else:
            correct_char = token.upper()
        score = 0.0    
        next_char_pred = lm.state_update(list(context))
        p = next_char_pred[[c[0] for c in next_char_pred].index(correct_char)][1]
        score = log10(p)
        print(f"p( {token} | {prev} ...) = {p:.6f} [ {score:.6f} ]")
        accum += score
        prev = token
        context += token
    print(f"sum logprob = {accum:.4f}, per-symbol logprob = {accum / symbols:.4f}, ppl = {pow(10,-1 * accum / symbols):.4f}\n")


    print("\nmodel = Uniform\n")

    lm = UniformLanguageModel()
    accum = 0.0

    prev = "<s>"
    context = ""

    for i,token in enumerate(tokens):
        correct_char = ""
        if(token == "<sp>"):
            token = "_"
            correct_char = "_"
        else:
            correct_char = token.upper()
        score = 0.0    
        next_char_pred = lm.predict(evidence=list(context))
        p = next_char_pred[[c[0] for c in next_char_pred].index(correct_char)][1]
        score = log10(p)
        print(f"p( {token} | {prev} ...) = {p:.6f} [ {score:.6f} ]")
        accum += score
        prev = token
        context += token
    print(f"sum logprob = {accum:.4f}, per-symbol logprob = {accum / symbols:.4f}, ppl = {pow(10,-1 * accum / symbols):.4f}\n")
