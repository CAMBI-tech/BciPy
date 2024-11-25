from bcipy.language.model.mixture import MixtureLanguageModel
from bcipy.core.symbols import alphabet
from bcipy.language.main import ResponseType


if __name__ == "__main__":
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    lm = MixtureLanguageModel(response_type, symbol_set)

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
