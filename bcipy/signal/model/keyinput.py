from typing import List


def compute_keyinput_probs(
    inquiry: List[str],
    symbol_set: List[str],
    user_error_prob: float,
    proceed: bool,
):
    """
    inquiry - the symbols presented in the inquiry preview
    symbol_set - the full alphabet of symbols
    proceed - whether or not the user wants to proceed with this inquiry
    user_error_prob - the probability that the user selects incorrectly during preview
    """
    K = len(inquiry)
    A = len(symbol_set)
    if not 0 <= user_error_prob <= 1:
        raise ValueError(f"invalid user error prob: {user_error_prob}")

    if proceed:  # user liked the inquiry; presented letters are upgraded and others are downgraded
        shown_letter_value = (1 - user_error_prob) / K
        other_letter_value = user_error_prob / (A - K)

    else:  # user disliked the inquiry; presented are downgraded and others are upgraded
        shown_letter_value = user_error_prob / K
        other_letter_value = (1 - user_error_prob) / (A - K)

    return {"KEYINPUT": [shown_letter_value if letter in inquiry else other_letter_value for letter in symbol_set]}
