from typing import List, Tuple

from bcipy.helpers.symbols import BACKSPACE_CHAR


def next_target_letter(current_sentence, target_sentence):
    """Computes the next target letter based on the currently spelled_text.
    """
    if target_sentence[0:len(current_sentence)] == current_sentence:
        # if correctly spelled so far, get the next letter.
        return target_sentence[len(current_sentence)] if len(
            current_sentence) < len(target_sentence) else target_sentence[-1]
    return BACKSPACE_CHAR


def format_lm_output(lm_evidence_tuples: List[Tuple], symbol_set):
    """ Formats language model output into list of likelihoods in alphabet order """
    pos_map = {letter: i for i, letter in enumerate(symbol_set)}
    reshaped_lm_lik = list(symbol_set)
    # sorting the likelhoods by order of symbol set
    for letter, lik in lm_evidence_tuples:
        reshaped_lm_lik[pos_map[letter]] = float(lik)

    return reshaped_lm_lik
