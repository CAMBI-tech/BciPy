import numpy as np
import logging
from typing import Dict, List
from copy import copy

log = logging.getLogger(__name__)


# Criteria
class DecisionCriteria:
    """Abstract class for Criteria which can be applied to evaluate a sequence
    """

    def __init__(self, **kwargs):
        pass

    def reset(self):
        pass

    def decide(self, epoch):
        """
        Apply the given criteria.
        Parameters:
        -----------
            epoch - Epoch data
                - target(str): target of the epoch
                - time_spent(ndarray[float]): |num_trials|x1
                      time spent on the sequence
                - list_sti(list[list[str]]): presented symbols in each
                      sequence
                - list_distribution(list[ndarray[float]]): list of |alp|x1
                        arrays with prob. dist. over alp

        """
        raise NotImplementedError()


class MinIterationsCriteria(DecisionCriteria):
    """ Returns true if the minimum number of iterations have not yet
        been reached. """

    def __init__(self, min_num_seq):
        """ Args:
                min_num_seq(int): minimum number of sequence number before any
                 termination objective is allowed to be triggered """
        self.min_num_seq = min_num_seq

    def decide(self, epoch):
        # Note: we use 'list_sti' parameter since this is the number of
        # sequences displayed. The length of 'list_distribution' is 1 greater
        # than this, since the language model distribution is added before
        # the first sequence is displayed.
        current_seq = len(epoch['list_sti'])
        log.debug(
            f"Checking min iterations; current iteration is {current_seq}")
        return current_seq < self.min_num_seq


class DecreasedProbabilityCriteria(DecisionCriteria):
    """Returns true if the letter with the max probability decreased from the
        last sequence."""

    def decide(self, epoch):
        if len(epoch['list_distribution']) < 2:
            return False
        prev_dist = epoch['list_distribution'][-2]
        cur_dist = epoch['list_distribution'][-1]
        return np.argmax(cur_dist) == np.argmax(
            prev_dist) and np.max(cur_dist) < np.max(prev_dist)


class MaxIterationsCriteria(DecisionCriteria):
    """Returns true if the max iterations have been reached."""

    def __init__(self, max_num_seq):
        """ Args:
                max_num_seq(int): maximum number of sequences allowed before
                    mandatory termination """
        self.max_num_seq = max_num_seq

    def decide(self, epoch):
        # Note: len(epoch['list_sti']) != len(epoch['list_distribution'])
        # see MinIterationsCriteria comment
        current_seq = len(epoch['list_sti'])
        if current_seq >= self.max_num_seq:
            log.debug(
                "Committing to decision: max iterations have been reached.")
            return True
        return False


class ProbThresholdCriteria(DecisionCriteria):
    """Returns true if the commit threshold has been met."""

    def __init__(self, threshold):
        """ Args:
                threshold(float in [0,1]): A threshold on most likely
                    candidate posterior. If a candidate exceeds a posterior
                    the system terminates.
                 """
        assert 1 >= threshold >= 0, "stopping threshold should be in [0,1]"
        self.tau = threshold

    def decide(self, epoch):
        current_distribution = epoch['list_distribution'][-1]
        if np.max(current_distribution) > self.tau:
            log.debug("Committing to decision: posterior exceeded threshold.")
            return True
        return False


class MarginCriteria(DecisionCriteria):
    """ Stopping criteria based on the difference of two most likely candidates.
        This condition
    """

    def __init__(self, margin):
        """ Args:
                margin(float in [0,1]): Minimum distance required between
                    two most likely competing candidates to trigger termination.
                    """
        assert 1 >= margin >= 0, "difference margin should be in [0,1]"
        self.margin = margin

    def decide(self, epoch):
        # Get the current posterior probability values
        p = copy(epoch['list_distribution'][-1])
        # This criteria compares most likely candidates (best competitors)
        candidates = [p[idx] for idx in list(np.argsort(p)[-2:])]
        stopping_rule = np.abs(candidates[0] - candidates[1])
        d = stopping_rule > self.margin
        if d:
            log.debug("Committing to decision: margin is high enough.")

        return d


class MomentumCommitCriteria(DecisionCriteria):
    """ Stopping criteria based on Shannon entropy on the simplex
        Attr:
            lam(float): linear combination parameter between entropy and the
                speed term
            tau(float): decision threshold
            """

    def __init__(self, tau, lam):
        self.lam = lam
        self.tau = tau

    def reset(self):
        pass

    def decide(self, epoch):
        eps = np.power(.1, 6)

        prob_history = copy(epoch['list_distribution'])
        p = prob_history[-1]

        tmp_p = np.ones(len(p)) * (1 - self.tau) / (len(p) - 1)
        tmp_p[0] = self.tau

        tmp = -np.sum(p * np.log2(p + eps))
        tau_ = -np.sum(tmp_p * np.log2(tmp_p + eps))

        tmp_ = np.array(prob_history)
        mom_ = tmp_[1:] * (
                np.log(tmp_[1:] + eps) - np.log(tmp_[:-1] + eps))
        momentum = np.linalg.norm(mom_) / len(prob_history)

        if len(prob_history) > 1:
            stopping_rule = tmp - self.lam * momentum
        else:
            stopping_rule = tmp

        d = stopping_rule < tau_

        return d


class CriteriaEvaluator():
    """Evaluates whether an epoch should commit to a decision based on the
    provided criteria.

    Parameters:
    -----------
        continue_criteria: list of criteria; if any of these evaluate to true the
            decision maker continues.
        commit_criteria: list of criteria; if any of these return true and
            continue_criteria are all false, decision maker commits to a decision.
    """

    def __init__(self, continue_criteria: List[DecisionCriteria],
                 commit_criteria: List[DecisionCriteria]):
        self.continue_criteria = continue_criteria or []
        self.commit_criteria = commit_criteria or []

    @classmethod
    def default(cls, min_num_seq, max_num_seq, threshold):
        return cls(continue_criteria=[MinIterationsCriteria(min_num_seq)],
                   commit_criteria=[
                       MaxIterationsCriteria(max_num_seq),
                       ProbThresholdCriteria(threshold)
                   ])

    def do_epoch(self):
        for el_ in self.continue_criteria:
            el_.reset()
        for el in self.commit_criteria:
            el.reset()

    def should_commit(self, epoch: Dict):
        """Evaluates the given epoch; returns true if stoppage criteria has
        been met, otherwise false.

        Parameters:
        -----------
            epoch - Epoch data
                - target(str): target of the epoch
                - time_spent(ndarray[float]): |num_trials|x1
                      time spent on the sequence
                - list_sti(list[list[str]]): presented symbols in each
                      sequence
                - list_distribution(list[ndarray[float]]): list of |alp|x1
                        arrays with prob. dist. over alp
        """
        if any(
                criteria.decide(epoch)
                for criteria in self.continue_criteria):
            return False
        return any(
            criteria.decide(epoch) for criteria in self.commit_criteria)
