import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from bcipy.data.stimuli import best_selection


class StimuliAgent(ABC):
    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def return_stimuli(self, list_distribution: np.ndarray, **kwargs):
        """ updates the agent with most likely posterior and selects queries
            Args:
                list_distribution(list[ndarray]): posterior distributions as
                    stored in the decision maker
            Return:
                query(list[str]): queries """
        ...

    @abstractmethod
    def do_series(self):
        """ If the system decides on a class let the agent know about it """
        ...


class RandomStimuliAgent(StimuliAgent):
    """ An inherited class of StimuliAgent. Chooses random set of letters for
        queries instead of most likely letters.
        Attr:
            alphabet(list[str]): Query space(possible queries).
            len_query(int): number of elements in a query
        Functions:
            reset(): reset the agent
            return_stimuli(): update the agent and return a stimuli set
            do_series(): one a commitment is made update agent
            """

    def __init__(self, alphabet: List[str], len_query: int = 4):
        self.alphabet = alphabet
        self.len_query = len_query

    def reset(self):
        """ This querying method is memoryless no reset needed """
        pass

    def return_stimuli(self, list_distribution: np.ndarray, constants: Optional[List[str]] = None):
        """ return random elements from the alphabet """
        tmp = [i for i in self.alphabet]
        query = random.sample(tmp, self.len_query)

        if constants:
            query[-len(constants):] = constants

        return query

    def do_series(self):
        pass


class NBestStimuliAgent(StimuliAgent):
    """ An inherited class of StimuliAgent. Updates the agent with N most likely
        letters based on posteriors and selects queries.
        Attr:
            alphabet(list[str]): Query space(possible queries).
            len_query(int): number of elements in a query
        Functions:
            reset(): reset the agent
            return_stimuli(): update the agent and return a stimuli set
            do_series(): one a commitment is made update agent
            """

    def __init__(self, alphabet: List[str], len_query: int = 4):
        self.alphabet = alphabet
        self.len_query = len_query

    def reset(self):
        pass

    def return_stimuli(self,
                       list_distribution: np.ndarray,
                       constants: Optional[List[str]] = None) -> List[str]:
        """Returns a list of the n most likely symbols based on the provided
        probabilities, where n is self.len_query. Symbols of the same
        probability will be ordered randomly.

        Parameters
        ----------
            list_distribution - list of lists of probabilities. Only the last list will
                be used.
            constants - optional list of symbols which should appear every result
        """
        symbol_probs = list(zip(self.alphabet, list_distribution[-1]))
        randomized = random.sample(symbol_probs, len(symbol_probs))
        symbols, probs = zip(*randomized)
        return best_selection(selection_elements=list(symbols),
                              val=list(probs),
                              len_query=self.len_query,
                              always_included=constants)

    def do_series(self):
        pass
