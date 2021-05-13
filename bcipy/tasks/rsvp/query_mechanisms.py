import numpy as np
from copy import copy
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


class MomentumStimuliAgent(StimuliAgent):
    # TODO - either add demo, add test, or delete this unused code!
    """ A query agent that utilizes the observed evidence so far at each step.
        This agent is specifically designed to overcome the adversary effect of
        the prior information to the system.
        There are two competing terms;
            I: mutual information, this utilizes the current posterior
            M: momentum, this utilizes all likelihoods so far
        and the queries are selected by linearly combining these two.
        This agent, in the beginning of the series, explores utilizing M and
        eventually exploits the knowledge using I  """

    def __init__(self, alphabet: List[str],
                 len_query: int = 4,
                 lam: float = 1.,
                 gam: float = 1.,
                 dif_lam: float = .09,
                 update_lam_flag: bool = True):
        """ query = max(I + lam * gam * M)
            lam(float): convex mixer term
            gam(float): the scaling factor, likelihoods are in log domain,
                whereas the mutual information term is not
            # TODO: mathematically this gam term is not required anymore.
            dif_lam(float): at each inquiry, you prefer to decrease the effect
                of M, otherwise you are only prune to noise
            update_lam_flag(bool): if True updates lambda at each inquiry
        """
        self.alphabet = alphabet
        self.lam = lam
        self.lam_ = copy(self.lam)
        self.dif_lam = dif_lam
        self.gam = np.float(gam)
        self.len_query = len_query
        self.update_lam_flag = update_lam_flag

    def reset(self):
        """ resets the history related items in the query agent """
        if self.update_lam_flag:
            self.lam_ = copy(self.lam)

    def return_stimuli(self, list_distribution: np.ndarray):
        """ Return the speed enhanced momentum stimuli """

        # To compute the MI term, use the most recent distribution
        tmp = list_distribution[-1]

        # conditional entropy as a surrogate for mutual information
        entropy_term = np.array(tmp) * np.log(tmp + 1e-6) + (
            1.01 - np.array(tmp)) * (np.log(1.01 - np.array(tmp)))
        entropy_term[np.isnan(entropy_term)] = 0

        # update the momentum term using likelihoods
        momentum = self._compute_momentum(list_distribution)
        num_passed_inquiries = len(list_distribution)

        # if there are no inquiries shown yet, the momentum cannot be computed
        if num_passed_inquiries > 1:
            reward = (self.lam_ - 1) * entropy_term + (
                self.lam_ / num_passed_inquiries) * momentum
        else:
            reward = -entropy_term

        # if lambda update flag is True, update the lambda
        if self.update_lam_flag:
            self._update_lam(len(list_distribution))

        tmp_alp = copy(self.alphabet)
        stimuli = best_selection(tmp_alp, reward, self.len_query)

        return stimuli

    def _compute_momentum(self, list_distribution):
        len_history = len(list_distribution)
        if len_history >= 2:
            tmp_ = len_history - np.arange(1, len_history)
            decay_scale = np.power(self.gam, tmp_)

            momentum_ = np.sum([decay_scale[k] * (list_distribution[k + 1] -
                                                  list_distribution[k])
                                for k in range(len_history - 1)], axis=0)
        else:
            momentum_ = 0

        return momentum_

    def _update_lam(self, len_history):
        """ Handles the handshaking between two objectives.
            currently just a constant shift with number of queries,
             should be updated logically """
        thr = 1
        if len_history < 10:
            # if less then 10 inquiries so far, do the hand shaking
            self.lam_ = np.max(
                [self.lam_ - self.dif_lam * (len_history / thr), 0])
        else:
            # do not explore if already passed 10 inquiries
            self.lam_ = 0

    def do_series(self):
        self.reset()


def best_selection(list_el: List[Any], val: List[float], len_query: int):
    """Return the top `len_query` items from `list_el` according to the values in `val`"""
    # numpy version: return list_el[(-val).argsort()][:len_query]
    sorted_items = reversed(sorted(zip(val, list_el)))
    return [el for (value, el) in sorted_items][:len_query]
