from bcipy.language.model.causal import CausalLanguageModelAdapter
from bcipy.core.symbols import DEFAULT_SYMBOL_SET


if __name__ == "__main__":
    lm = CausalLanguageModelAdapter(lang_model_name="figmtu/opt-350m-aac")
    lm.set_symbol_set(DEFAULT_SYMBOL_SET)

    print("Target sentence: does_it_make_sense\n")

    next_char_pred = lm.predict_character(list("does_it_make_sen"))
    print("Context: does_it_make_sen")
    print(f"Predictions: {next_char_pred}")
    correct_char_rank = [c[0] for c in next_char_pred].index("S") + 1
    print(f"Correct character rank: {correct_char_rank}\n")

    next_char_pred = lm.predict_character(list("does_it_make_sens"))
    print("Context: does_it_make_sens")
    print(f"Predictions: {next_char_pred}")
    correct_char_rank = [c[0] for c in next_char_pred].index("E") + 1
    print(f"Correct character rank: {correct_char_rank}\n")

    next_char_pred = lm.predict_character(list("does_it_make_sense"))
    print("Context: does_it_make_sense")
    print(f"Predictions: {next_char_pred}")
    correct_char_rank = [c[0] for c in next_char_pred].index("_") + 1
    print(f"Correct character rank: {correct_char_rank}\n")

    print("Target sentence: i_like_zebras\n")

    next_char_pred = lm.predict_character(list("i_like_zebra"))
    print("Context: i_like_zebra")
    print(f"Predictions: {next_char_pred}")
    correct_char_rank = [c[0] for c in next_char_pred].index("S") + 1
    print(f"Correct character rank: {correct_char_rank}\n")
