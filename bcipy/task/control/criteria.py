import numpy as np
import logging
from typing import Dict, List
from copy import copy

log = logging.getLogger(__name__)


class DecisionCriteria:
    """Abstract class for Criteria which can be applied to evaluate a inquiry
    """

    def __init__(self, **kwargs):
        pass

    def reset(self):
        pass

    def decide(self, series: Dict):
        """
        Apply the given criteria.
        Parameters:
        -----------
            series - series data
                - target(str): target of the series
                - time_spent(ndarray[float]): |num_trials|x1
                      time spent on the inquiry
                - list_sti(list[list[str]]): presented symbols in each
                      inquiry
                - list_distribution(list[ndarray[float]]): list of |alp|x1
                        arrays with prob. dist. over alp

        """
        raise NotImplementedError()


class MinIterationsCriteria(DecisionCriteria):
    """ Returns true if the minimum number of iterations have not yet
        been reached. """

    def __init__(self, min_num_inq: int):
        """ Args:
                min_num_inq(int): minimum number of inquiry number before any
                 termination objective is allowed to be triggered """
        self.min_num_inq = min_num_inq

    def decide(self, series: Dict):
        # Note: we use 'list_sti' parameter since this is the number of
        # inquiries displayed. The length of 'list_distribution' is 1 greater
        # than this, since the language model distribution is added before
        # the first inquiry is displayed.
        current_inq = len(series['list_sti'])
        log.debug(
            f"Checking min iterations; current iteration is {current_inq}")
        return current_inq < self.min_num_inq


class DecreasedProbabilityCriteria(DecisionCriteria):
    """Returns true if the letter with the max probability decreased from the
        last inquiry."""

    def decide(self, series: Dict):
        if len(series['list_distribution']) < 2:
            return False
        prev_dist = series['list_distribution'][-2]
        cur_dist = series['list_distribution'][-1]
        return np.argmax(cur_dist) == np.argmax(
            prev_dist) and np.max(cur_dist) < np.max(prev_dist)


class MaxIterationsCriteria(DecisionCriteria):
    """Returns true if the max iterations have been reached."""

    def __init__(self, max_num_inq: int):
        """ Args:
                max_num_inq(int): maximum number of inquiries allowed before
                    mandatory termination """
        self.max_num_inq = max_num_inq

    def decide(self, series: Dict):
        # Note: len(series['list_sti']) != len(series['list_distribution'])
        # see MinIterationsCriteria comment
        current_inq = len(series['list_sti'])
        if current_inq >= self.max_num_inq:
            log.debug(
                "Committing to decision: max iterations have been reached.")
            return True
        return False


class ProbThresholdCriteria(DecisionCriteria):
    """Returns true if the commit threshold has been met."""

    def __init__(self, threshold: float):
        """ Args:
                threshold(float in [0,1]): A threshold on most likely
                    candidate posterior. If a candidate exceeds a posterior
                    the system terminates.
                 """
        assert 1 >= threshold >= 0, "stopping threshold should be in [0,1]"
        self.tau = threshold

    def decide(self, series: Dict):
        current_distribution = series['list_distribution'][-1]
        if np.max(current_distribution) > self.tau:
            log.debug("Committing to decision: posterior exceeded threshold.")
            return True
        return False


class MarginCriteria(DecisionCriteria):
    """ Stopping criteria based on the difference of two most likely candidates.
        This condition enforces the likelihood difference between two top
        candidates to be at least a value. E.g. in 4 category case with
        a margin 0.2, the edge cases [0.6,0.4,0.,0.] and [0.4,0.2,0.2,0.2]
        satisfy the condition.
    """

    def __init__(self, margin: float):
        """ Args:
                margin(float in [0,1]): Minimum distance required between
                    two most likely competing candidates to trigger termination.
                    """
        assert 1 >= margin >= 0, "difference margin should be in [0,1]"
        self.margin = margin

    def decide(self, series: Dict):
        # Get the current posterior probability values
        p = copy(series['list_distribution'][-1])
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

    def __init__(self, tau: float, lam: float):
        self.lam = lam
        self.tau = tau

    def reset(self):
        pass

    def decide(self, series):
        eps = np.power(.1, 6)

        prob_history = copy(series['list_distribution'])
        p = prob_history[-1]

        tmp_p = np.ones(len(p)) * (1 - self.tau) / (len(p) - 1)
        tmp_p[0] = self.tau

        tmp = -np.sum(p * np.log2(p + eps))
        tau_ = -np.sum(tmp_p * np.log2(tmp_p + eps))

        tmp_ = np.array(prob_history)
        mom_ = tmp_[1:] * (np.log(tmp_[1:] + eps) - np.log(tmp_[:-1] + eps))
        momentum = np.linalg.norm(mom_) / len(prob_history)

        if len(prob_history) > 1:
            stopping_rule = tmp - self.lam * momentum
        else:
            stopping_rule = tmp

        d = stopping_rule < tau_

        return d


class CriteriaEvaluator():
    """Evaluates whether an series should commit to a decision based on the
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
    def default(cls, min_num_inq: int, max_num_inq: int, threshold: float):
        return cls(continue_criteria=[MinIterationsCriteria(min_num_inq)],
                   commit_criteria=[
                       MaxIterationsCriteria(max_num_inq),
                       ProbThresholdCriteria(threshold)
        ])

    def do_series(self):
        for el_ in self.continue_criteria:
            el_.reset()
        for el in self.commit_criteria:
            el.reset()

    def should_commit(self, series: Dict):
        """Evaluates the given series; returns true if stoppage criteria has
        been met, otherwise false.

        Parameters:
        -----------
            series - series data
                - target(str): target of the series
                - time_spent(ndarray[float]): |num_trials|x1
                      time spent on the inquiry
                - list_sti(list[list[str]]): presented symbols in each
                      inquiry
                - list_distribution(list[ndarray[float]]): list of |alp|x1
                        arrays with prob. dist. over alp
        """
        if any(
                criteria.decide(series)
                for criteria in self.continue_criteria):
            return False
        return any(
            criteria.decide(series) for criteria in self.commit_criteria)
