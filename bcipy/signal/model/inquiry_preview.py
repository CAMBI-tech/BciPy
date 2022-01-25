from typing import List


def compute_probs_after_preview(
    query: List[str], alphabet: List[str], user_error_prob: float, proceed: bool
) -> List[float]:
    r"""
    Notation:
    - "want" = desired letter
    - "show" = shown letters
    - "button" = user's button response. "+" = user selected, "-" = user not selected
    - "past" = prior from language model and possibly previous button evidence

    Assumptions:
    1. User's errors are symmetric. p(+ | want not in show) = p(- | want in show) = user_error_prob
       This lets us use a single error rate to describe the user behavior for all cases
       (desired letter shown/not shown, response of + or -)
       The alternative would be determining one rate of accidental presses and another rate of accidental misses.

    2. Given the shown letters and user's desired letter, their response is independent of previous events.
       The user's behavior is fully determined by their desired letter and the shown letters.
       Only some edge effects such as frustration or fatigue are ignored by this.

    We're interested in the alphabet posterior overall:

        p(want | button, show, past) = p(button | want, show, past) p(want | show, past) / p(button | show, past)
                                    # Use assumption 2.
                                    = p(button | want, show) p(want | show, past) / p(button | show, past)
                                    # Given past, show does not affect want
                                    = p(button | want, show) p(want | past) / p(button | show, past)
                                    # Will normalize over alphabet anyway. Can ignore normalization const.
                                    \propto p(button | want, show) p(want | past)
                                    # Leave out the prior - this is handled outside by evidence fusion.
                                    \propto p(button | want, show)
                                            ~~~~~~~~~~~~~~~~~~~~~~
                                                    |
                                    These are the multiplicative updates we'll return.


    For shown letters, if the user gives a positive button response, we have terms like:

        p(want=A | +, show=ABC, past) \propto p(+ | want=A, show=ABC)
                                      \propto p(+ | want in show) = "probability of correct button press"

    And for shown letters, a negative response gives us terms like this:

        p(want=A | -, show=ABC, past) \propto p(- | want=A, show=ABC)
                                      \propto p(- | want in show) = "probability of incorrect button press"

    For unshown letter, positive button response:

        p(want=Z | +, show=ABC, past) \propto p(+ | want not in show) = "probability of incorrect button press"

    For unshown letter, negative button response:

        p(want=Z | -, show=ABC, past) \propto p(- | want not in show) = "probability of correct button press"

    Note that this simple rule, followed by normalizing the result vector, has the desired scaling property;
    with larger inquiries, the prior probability of a "+" reponse is larger.
    Thus, for a "+" response, the total probability given to the shown letters should grow as the inquiry length grows.

    >>> def example(N, L):
    ...     # Show N letters from an alphabet of length L
    ...     probs = np.array([0.95]*N + [0.05] * (L-N))
    ...     probs /= probs.sum()
    ...     # Check the effect of a "+" response on the N shown letters
    ...     res = probs[:N].sum()
    ...     print(format(res, "0.2f"))
    >>> example(2, 10)
    0.83
    >>> example(5, 10)
    0.95
    >>> example(8, 10)
    0.99

    Args:
        query - the symbols presented in the query preview
        alphabet - the full alphabet of symbols
        proceed - whether or not the user wants to proceed with this query
        user_error_prob - the probability that the user selects incorrectly during preview
    """
    if not 0 <= user_error_prob <= 1:
        raise ValueError(f"invalid user error prob: {user_error_prob}")

    if proceed:
        shown_letter_value = 1 - user_error_prob
        other_letter_value = user_error_prob
    else:
        shown_letter_value = user_error_prob
        other_letter_value = 1 - user_error_prob

    return [shown_letter_value if letter in query else other_letter_value for letter in alphabet]
