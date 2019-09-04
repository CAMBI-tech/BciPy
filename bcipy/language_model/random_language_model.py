"""Language Model that produces random probabilities"""

import logging
from typing import List

import numpy as np

class RandomLm:
    """Language Model that produces random likelihoods."""

    def __init__(self, alphabet):

        self.log = logging.getLogger(__name__)
        self.log.debug("Using Random Language Model")
        self.normalized = True  # normalized to the probability domain.
        self.alp = alphabet

    def state_update(self, evidence: List, return_mode: str = 'letter'):
        """
        Provide a prior distribution of the language model.

        Input:
            evidence - list of current evidence
            return_mode - desired return mode
        Output:
            priors - a json dictionary with Normalized priors
                     in the Negative Log probability domain.
        """
        sample = uniform(len(self.alp))
        pairs = list(zip(self.alp, sample.tolist()))
        priors = {return_mode: pairs}

        self.log.debug("Language Model Random probabilities:")
        self.log.debug(priors)
        return priors


def uniform(n_letters, delta=0.0001):
    """Generate a uniform distribution.

    1. Sum to 1.0
    2. Have length n_letters and will all be different values
    3. Values will be close to equally probable within +- delta.

    The resulting list is used to create a sort order while only minimally
    affecting the default probability value.
    """
    equal_prob = 1 / n_letters
    result = np.random.uniform(low=equal_prob - delta,
                               high=equal_prob + delta,
                               size=n_letters)

    # Ensure that all values are different; it's not clear whether numpy
    # guarantees this property.
    while len(set(result)) != len(result):
        result = np.random.uniform(low=equal_prob - delta,
                                   high=equal_prob + delta,
                                   size=n_letters)
    # return normalized values
    return result / np.sum(result)
