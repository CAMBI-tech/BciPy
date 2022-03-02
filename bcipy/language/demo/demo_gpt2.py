from bcipy.language.model.gpt2 import GPT2LanguageModel
from bcipy.helpers.task import alphabet
from bcipy.language.main import ResponseType


if __name__ == "__main__":
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    lm = GPT2LanguageModel(response_type, symbol_set)

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

    '''
    next_char_pred = lm.state_update(list("PEANUT_BUTTER_AND_IT_"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("PEANUT_BUTTER_AND_"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("DO_WE_HAVE_ANY_CH"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("DO_WE_HAVE_ANY_CHIPS"))
    print(next_char_pred)
    '''

    '''
    next_char_pred = lm.state_update(list("PEANUT_BUTTER_AND_"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("PEANUT_BUTTER_AND_J"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("PEANUT_BUTTER_AND_JELLY"))
    print(next_char_pred)
    '''

    '''
    next_char_pred = lm.state_update(list("AS_SOON_AS_"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("AS_SOON_AS_P"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("AS_SOON_AS_PO"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("AS_SOON_AS_POS"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("AS_SOON_AS_POSS"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("AS_SOON_AS_POSSI"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("AS_SOON_AS_POSSIB"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("AS_SOON_AS_POSSIBL"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("AS_SOON_AS_POSSIBLE"))
    print(next_char_pred)
    next_char_pred = lm.state_update(list("AS_SOON_AS_POSSIBLE_"))
    print(next_char_pred)
    '''
