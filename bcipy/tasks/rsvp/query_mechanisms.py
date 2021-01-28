import numpy as np
from copy import copy
import random
from typing import List

# the small epsilon value. Prevents numerical issues with log
eps = np.power(.1, 6)


class StimuliAgent:
    """ Query mechanism base class.
        Attr:
            alphabet(list[str]): Query space(possible queries).
            len_query(int): number of elements in a query
        Functions:
            reset(): reset the agent
            return_stimuli(): update the agent and return a stimuli set
            do_series(): one a commitment is made update agent
            """

    def __init__(self,
                 alphabet: List[str],
                 len_query: int,
                 **kwargs):
        self.alp = alphabet
        self.len_query = len_query

    def reset(self):
        return

    def return_stimuli(self, list_distribution: np.ndarray, **kwargs):
        """ updates the agent with most likely posterior and selects queries
            Args:
                list_distribution(list[ndarray]): posterior distributions as
                    stored in the decision maker
            Return:
                query(list[str]): queries """
        return

    def do_series(self):
        """ If the system decides on a class let the agent know about it """
        return


class RandomStimuliAgent(StimuliAgent):
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
        entropy_term = np.array(tmp) * np.log(tmp + eps) + (
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


# A generic best selection from set function using values
def best_selection(list_el, val, len_query):
    """ given set of elements and a value function over the set,
        picks the len_query number of elements with the best value.
        Args:
            list_el(list[str]): the set of elements
            val(list[float]): values for the corresponding elements
            len_query(int): number of elements to be picked from the set
        Return:
            query(list[str]): elements from list_el with the best value """
    max_p_val = np.sort(val)[::-1]
    max_p_val = max_p_val[0:len_query]

    query = []
    for idx in range(len_query):
        idx_q = np.where(val == max_p_val[idx])[0][0]

        q = list_el[idx_q]
        val = np.delete(val, idx_q)
        list_el.remove(q)
        query.append(q)

    return query
