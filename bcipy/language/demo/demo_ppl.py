from bcipy.language.model.gpt2 import GPT2LanguageModel
from bcipy.helpers.task import alphabet
from bcipy.language.main import ResponseType
from math import log10

if __name__ == "__main__":
    
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    lm = GPT2LanguageModel(response_type, symbol_set)

    
    sentence = "i <sp> l i k e <sp> z e b r a s"
    accum = 0.0

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
        # print(next_char_pred)
        p = next_char_pred[[c[0] for c in next_char_pred].index(correct_char)][1]
        score = log10(p)
        print(f"p( {token} | {prev} ...) = {p:.6f} [ {score:.6f} ]")
        accum += score
        prev = token
        context += token
    print(f"sum logprob = {accum:.4f}, per-symbol logprob = {accum / symbols:.4f}, ppl = {pow(10,-1 * accum / symbols):.4f}")