from typing import List


def compute_probs_after_preview(
    inquiry: List[str], symbol_set: List[str], user_error_prob: float, proceed: bool
) -> List[float]:
    """
    When the user responds to an inquiry preview, we compute a likelihood term for each symbol
    based on their response. These likelihood terms will be multiplied with the prior probabilities of
    each symbol and then normalized, in order to update the symbol probabilities.

    Args:
        inquiry - the symbols presented in the inquiry preview
        symbol_set - the full alphabet of symbols
        proceed - whether or not the user wants to proceed with this inquiry
        user_error_prob - the probability that the user selects incorrectly during preview
    """
    inq_len = len(inquiry)
    sym_set_len = len(symbol_set)
    if not 0 <= user_error_prob <= 1:
        raise ValueError(f"invalid user error prob: {user_error_prob}")

    if proceed:  # user liked the inquiry; presented letters are upgraded and others are downgraded
        shown_letter_value = (1 - user_error_prob) / inq_len
        other_letter_value = user_error_prob / (sym_set_len - inq_len)

    else:  # user disliked the inquiry; presented are downgraded and others are upgraded
        shown_letter_value = user_error_prob / inq_len
        other_letter_value = (1 - user_error_prob) / (sym_set_len - inq_len)

    return [shown_letter_value if letter in inquiry else other_letter_value for letter in symbol_set]
