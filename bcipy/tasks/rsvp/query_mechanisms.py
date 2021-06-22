import numpy as np
import random
from typing import List, Any
from abc import ABC, abstractmethod


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

    def return_stimuli(self, list_distribution: np.ndarray):
        """ return random elements from the alphabet """
        tmp = [i for i in self.alphabet]
        query = random.sample(tmp, self.len_query)

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

    def return_stimuli(self, list_distribution: np.ndarray):
        p = list_distribution[-1]
        tmp = [i for i in self.alphabet]
        query = best_selection(tmp, p, self.len_query)

        return query

    def do_series(self):
        pass


def best_selection(list_el: List[Any], val: List[float], len_query: int):
    """Return the top `len_query` items from `list_el` according to the values in `val`"""
    # numpy version: return list_el[(-val).argsort()][:len_query]
    sorted_items = reversed(sorted(zip(val, list_el)))
    return [el for (value, el) in sorted_items][:len_query]
