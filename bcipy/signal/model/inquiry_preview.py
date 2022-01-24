from typing import List


def _OLD_compute_probs_after_preview(
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


def compute_probs_after_preview(
    query: List[str], alphabet: List[str], user_error_prob: float, proceed: bool
) -> List[float]:
    """
    Notation:
    - "want" = desired letter
    - "show" = shown letters
    - "+" = user selected
    - "-" = user not selected

    Assumptions:
    1. "want" independent of "show".
        - if we start from uniform alphabet, and treat each query as independent event, this is true
        - this is false if our shown letters are very enriched (in that case, we would bet on the displayed letters!)
        - used to say p(want | show) = p(want)
        - To avoid this assumption, we need info on past queries and responses somehow!
    2. User's errors are symmetric. p(+ | want not in show) = p(- | want in show) = user_error_prob

    Args:
        query - the symbols presented in the query preview
        alphabet - the full alphabet of symbols
        proceed - whether or not the user wants to proceed with this query
        user_error_prob - the probability that the user selects incorrectly during preview
    """
    if not 0 <= user_error_prob <= 1:
        raise ValueError(f"invalid user error prob: {user_error_prob}")

    """suppose user_error_prob = 0.05, and len(alphabet) = 28. suppose we present AB (random 2 of 28 letters).
    the user responds positively ("+")
        p(want=A | +, show=AB) = p(+ | want=A, show=AB) p(want=A | show=AB) / p(+ | show=AB)
                                = p(+ | want in show) p(want=A) / p(+ | show=AB)                 # Use assumption 1
                                = 0.95 * (1/28) / [
                                    p(+ | want=A, show=AB) p(want=A)
                                    + p(+ | want=B, show=AB) p(want=B)
                                    + p(+ | want=C, show=AB) p(want=C)
                                    ...
                                    + p(+ | want=Z, show=AB) p(want=Z)
                                ]

    We can compute this from just 2 terms:
        shown_term = p(+ | want=shown_letter, show=[....,shown_letter,...]) p(want=shown_letter)
        unshown_term = p(+ | want=unshown_letter, show=[...]) p(want=unshown_letter)

    Then we can update as follows:
        p(shown_letter | +, show=[...]) = shown_term / [
                                            len(show) * shown_term
                                            + (len(alphabet) - len(show)) * unshown_term
                                        ]

        p(unshown_letter | +, show=[...]) = unshown_term / [
                                            len(show) * shown_term
                                            + (len(alphabet) - len(show)) * unshown_term
                                        ]

    If user responds negatively ("-"):
    Same assumptions as above, but now user responds "-"
        p(want=A | -, show=AB) = p(- | want=A, show=AB) p(want=A | show=AB) / p(- | show=AB)
                                = p(- | want in show) p(want=A) / p(- | show=AB)                 # Use assumption 1

    We can compute this from just 2 terms:
        shown_term = p(- | want=shown_letter, show=[....,shown_letter,...]) p(want=shown_letter)
        unshown_term = p(- | want=unshown_letter, show=[...]) p(want=unshown_letter)

    Then we can update as follows:
        p(shown_letter | -, show=[...]) = shown_term / [
                                            len(show) * shown_term
                                            + (len(alphabet) - len(show)) * unshown_term
                                        ]

        p(unshown_letter | -, show=[...]) = unshown_term / [
                                            len(show) * shown_term
                                            + (len(alphabet) - len(show)) * unshown_term
                                        ]"""

    if proceed:
        shown_term = 1 - user_error_prob
        unshown_term = user_error_prob
    else:
        shown_term = user_error_prob
        unshown_term = 1 - user_error_prob

    # the denominator for bayes rule.
    # If inquiry is very long, the prior prob of selecting the inquiry is high.
    # Normalizing here accounts for this.
    n_shown = len(query)
    n_unshown = len(alphabet) - n_shown
    marginal = n_shown * shown_term + n_unshown * unshown_term

    # shown_letter_value = shown_term / marginal
    # other_letter_value = unshown_term / marginal
    shown_letter_value = shown_term
    other_letter_value = unshown_term

    return [shown_letter_value if letter in query else other_letter_value for letter in alphabet]
