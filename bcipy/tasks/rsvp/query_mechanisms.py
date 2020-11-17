import numpy as np
from copy import copy
import random

# the small epsilon value. Prevents numerical issues with log
eps = np.power(.1, 6)


class QueryAgent:
    """ Query mechanism base class.
        Attr:
            alphabet(list[str]): Query space(possible queries).
            len_query(int): number of elements in a query
        Functions:
            reset(): reset the agent
            update_and_query(): update the agent and return a query set
            do_epoch(): one a commitment is made update agent
            """

    def __init__(self, alphabet, len_query, **kwargs):
        self.alp = alphabet
        self.len_query = len_query

    def reset(self):
        return

    def update_and_query(self, p, **kwargs):
        """ updates the agent with most likely posterior and selects queries
            Args:
                p(list[float]): latest_posterior
            Return:
                query(list[str]): queries """
        return

    def do_epoch(self):
        """ If the system decides on a class let the agent know about it """
        return


class RandomAgent(QueryAgent):
    def __init__(self, alphabet, len_query=4):
        self.alphabet = alphabet
        self.len_query = len_query

    def reset(self):
        """ This querying method is memoryless no reset needed """
        pass

    def update_and_query(self, p):
        """ return random elements from the alphabet """
        tmp = [i for i in self.alphabet]
        query = random.sample(tmp, self.len_query)

        return query

    def do_epoch(self):
        pass


class NBestAgent(QueryAgent):
    def __init__(self, alphabet, len_query=4):
        self.alphabet = alphabet
        self.len_query = len_query

    def reset(self):
        pass

    def update_and_query(self, p):
        """ With the final belief over the system, updates the querying method
            and generates len_query most likely queries.
            Args:
                p(list[float]): list of probability distribution
            Return:
                query(list[str]): queries """
        tmp = [i for i in self.alphabet]
        query = best_selection(tmp, p, self.len_query)

        return query

    def do_epoch(self):
        pass


class MomentumQueryAgent(QueryAgent):
    """ A query agent that utilizes the observed evidence so far at each step.
        This agent is specifically designed to overcome the adversary effect of
        the prior information to the system. 
        There are two competing terms;
            I: mutual information, this utilizes the current posterior
            M: momentum, this utilizes all likelihoods so far
        and the queries are selected by linearly combining these two.
        This agent, in the beginning of the epoch, explores utilizing M and 
        eventually exploits the knowledge using I  """

    def __init__(self, alphabet, len_query=4, lam=1., gam=1., dif_lam=.09,
                 update_lam_flag=True):
        """ query = max(I + lam * gam * M)
            lam(float): convex mixer term
            gam(float): the scaling factor, likelihoods are in log domain,
                whereas the mutual information term is not
            # TODO: mathematically this gam term is not required anymore.
            dif_lam(float): at each sequence, you prefer to decrease the effect
                of M, otherwise you are only prune to noise
            update_lam_flag(bool): if True updates lambda at each sequence
        """
        self.alphabet = alphabet
        self.lam = lam
        self.lam_ = copy(self.lam)
        self.dif_lam = dif_lam
        self.gam = np.float(gam)
        self.len_query = len_query
        self.update_lam_flag = update_lam_flag
        self.momentum = np.zeros(len(self.alphabet))
        self.prob_history = []
        self.last_query = []

    def reset(self):
        """ resets the history related items in the query agent """
        if self.update_lam_flag:
            self.lam_ = copy(self.lam)

        self.momentum = np.zeros(len(self.alphabet))
        self.prob_history = []

    def update_and_query(self, p):
        """ Update the agent with the final belief over the system
            Observe that the agent has a memory of the past.
            Args:
                p(list[float]): list of probability distribution
                len_query(int): number of queries in the scheduled query
            Return:
                query(list[str]): queries """

        # The agent keeps a copy of the previous probabilities in the epoch
        tmp = p[:]
        self.prob_history.append(tmp)

        # update the momentum term using likelihoods
        self.update_momentum()
        num_passed_sequences = len(self.prob_history)

        # conditional entropy as a surrogate for mutual information
        entropy_term = np.array(tmp) * np.log(tmp + eps) + (
                1.01 - np.array(tmp)) * (np.log(1.01 - np.array(tmp)))
        entropy_term[np.isnan(entropy_term)] = 0

        # if there are no sequences shown yet, the momentum cannot be computed
        if num_passed_sequences > 1:
            reward = (self.lam_ - 1) * entropy_term + (
                    self.lam_ / num_passed_sequences) * self.momentum
        else:
            reward = -entropy_term

        # if lambda update flag is True, update the lambda
        if self.update_lam_flag:
            self.update_lam(len(self.prob_history))

        tmp_alp = copy(self.alphabet)
        tmp_query = best_selection(tmp_alp, reward, self.len_query)

        self.last_query = copy(tmp_query)

        return self.last_query

    def update_momentum(self):
        """ momentum is updated with the particular probability history.
            WARNING!: if called twice without a probability update, will update
            momentum using the same information twice """
        if len(self.prob_history) >= 2:
            # only update momentum for previous terms
            idx_prev_query = [self.alphabet.index(self.last_query[i]) for i
                              in range(len(self.last_query))]

            # scale the momentum value
            self.momentum *= self.gam

            for k in idx_prev_query:
                # momentum = current_mass * mass_displacement
                momentum_ = self.prob_history[-1][k] * (
                        np.log(self.prob_history[-1][k] + eps) - np.log(
                    self.prob_history[-2][k] + eps))
                self.momentum[k] += momentum_

    def update_lam(self, len_history):
        """ Handles the handshaking between two objectives.
            currently just a constant shift with number of queries,
             should be updated logically """
        thr = 1
        if len_history < 10:
            # if less then 10 sequences so far, do the hand shaking
            self.lam_ = np.max(
                [self.lam_ - self.dif_lam * (len_history / thr), 0])
        else:
            # do not explore if already passed 10 sequences
            self.lam_ = 0

    def do_epoch(self):
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
        try:
            idx_q = np.where(val == max_p_val[idx])[0][0]
        except:
            import pdb
            pdb.set_trace()

        q = list_el[idx_q]
        val = np.delete(val, idx_q)
        list_el.remove(q)
        query.append(q)

    return query
