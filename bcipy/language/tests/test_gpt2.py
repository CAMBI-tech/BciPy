from bcipy.language.model.gpt2 import GPT2LanguageModel
from bcipy.helpers.task import alphabet
from bcipy.language.main import ResponseType


if __name__ == "__main__":
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    lm = GPT2LanguageModel(response_type, symbol_set, "gpt2")

    next_char_pred = lm.state_update(list("does_it_make_sen"))
    print(next_char_pred)
    correct_char_rank = [c[0] for c in next_char_pred].index("S") + 1
    print(correct_char_rank)
    next_char_pred = lm.state_update(list("does_it_make_sens"))
    print(next_char_pred)
    correct_char_rank = [c[0] for c in next_char_pred].index("E") + 1
    print(correct_char_rank)
    next_char_pred = lm.state_update(list("does_it_make_sense"))
    print(next_char_pred)
    correct_char_rank = [c[0] for c in next_char_pred].index("_") + 1
    print(correct_char_rank)
