from bcipy.language.model.gpt2 import GPT2LanguageModel
from bcipy.language.main import alphabet
from bcipy.language.main import ResponseType
from timeit import default_timer as timer

if __name__ == "__main__":
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    lm = GPT2LanguageModel(response_type, symbol_set)
    start = timer()

    context = "i_like_"
    #context = "i_like_zebra"
    #context = "i_like_ze"
    #context = "my cat felix who is very fat, furry, and super cute is having a whole lot of fun chasing a m"

    next_char_pred = lm.state_update(list(context))
    end = timer()
    #print(f"prediction time = {(end-start):.4f}")

#    next_char_pred = lm.state_update(list("does_it_make_sen"))
#    print(next_char_pred)
#    correct_char_rank = [c[0] for c in next_char_pred].index("S") + 1
#    print(correct_char_rank)
#    next_char_pred = lm.state_update(list("does_it_make_sens"))
#    print(next_char_pred)
#    correct_char_rank = [c[0] for c in next_char_pred].index("E") + 1
#    print(correct_char_rank)
#    next_char_pred = lm.state_update(list("does_it_make_sense"))
#    print(next_char_pred)
#    correct_char_rank = [c[0] for c in next_char_pred].index("_") + 1
#    print(correct_char_rank)
