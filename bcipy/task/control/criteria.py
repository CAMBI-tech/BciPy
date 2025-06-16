"""Decision criteria module for BCI task control.

This module provides classes for evaluating decision criteria in BCI tasks.
These criteria are used to determine when to stop collecting evidence and
make a decision based on the accumulated data.
"""

import logging
from copy import copy
from typing import Dict, List, Any, Optional

import numpy as np

from bcipy.config import SESSION_LOG_FILENAME

log = logging.getLogger(SESSION_LOG_FILENAME)


class DecisionCriteria:
    """Abstract base class for decision criteria evaluation.

    This class defines the interface for criteria that can be applied to
    evaluate whether a decision should be made based on accumulated evidence.

    Attributes:
        None
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the decision criteria.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def reset(self) -> None:
        """Reset the criteria state."""
        pass

    def decide(self, series: Dict[str, Any]) -> bool:
        """Apply the decision criteria to the given series data.

        Args:
            series: Dictionary containing series data with the following keys:
                target (str): Target of the series.
                time_spent (np.ndarray): Time spent on each inquiry.
                list_sti (List[List[str]]): Presented symbols in each inquiry.
                list_distribution (List[np.ndarray]): Probability distributions
                    over the alphabet for each inquiry.

        Returns:
            bool: True if the criteria is met, False otherwise.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError()


class MinIterationsCriteria(DecisionCriteria):
    """Criteria for ensuring a minimum number of iterations.

    Returns true if the minimum number of iterations have not yet been reached.

    Attributes:
        min_num_inq: Minimum number of inquiries required.
    """

    def __init__(self, min_num_inq: int) -> None:
        """Initialize the minimum iterations criteria.

        Args:
            min_num_inq: Minimum number of inquiries required before any
                termination objective is allowed to be triggered.
        """
        self.min_num_inq = min_num_inq

    def decide(self, series: Dict[str, Any]) -> bool:
        """Check if minimum number of iterations has been reached.

        Note: Uses 'list_sti' parameter since this is the number of inquiries
        displayed. The length of 'list_distribution' is 1 greater than this,
        since the language model distribution is added before the first
        inquiry is displayed.

        Args:
            series: Dictionary containing series data.

        Returns:
            bool: True if current iterations < minimum required, False otherwise.
        """
        current_inq = len(series['list_sti'])
        log.info(f"Checking min iterations; current iteration is {current_inq}")
        return current_inq < self.min_num_inq


class DecreasedProbabilityCriteria(DecisionCriteria):
    """Criteria for detecting decreased probability of the most likely symbol.

    Returns true if the letter with the max probability decreased from the
    last inquiry.

    Attributes:
        None
    """

    def decide(self, series: Dict[str, Any]) -> bool:
        """Check if probability of most likely symbol has decreased.

        Args:
            series: Dictionary containing series data.

        Returns:
            bool: True if probability decreased for same symbol, False otherwise.
        """
        if len(series['list_distribution']) < 2:
            return False
        prev_dist = series['list_distribution'][-2]
        cur_dist = series['list_distribution'][-1]
        return np.argmax(cur_dist) == np.argmax(prev_dist) and np.max(cur_dist) < np.max(prev_dist)


class MaxIterationsCriteria(DecisionCriteria):
    """Criteria for enforcing maximum number of iterations.

    Returns true if the maximum allowed iterations have been reached.

    Attributes:
        max_num_inq: Maximum number of inquiries allowed.
    """

    def __init__(self, max_num_inq: int) -> None:
        """Initialize the maximum iterations criteria.

        Args:
            max_num_inq: Maximum number of inquiries allowed before
                mandatory termination.
        """
        self.max_num_inq = max_num_inq

    def decide(self, series: Dict[str, Any]) -> bool:
        """Check if maximum iterations have been reached.

        Note: len(series['list_sti']) != len(series['list_distribution'])
        See MinIterationsCriteria comment.

        Args:
            series: Dictionary containing series data.

        Returns:
            bool: True if max iterations reached, False otherwise.
        """
        current_inq = len(series['list_sti'])
        if current_inq >= self.max_num_inq:
            log.info("Committing to decision: max iterations have been reached.")
            return True
        return False


class ProbThresholdCriteria(DecisionCriteria):
    """Criteria for probability threshold-based decisions.

    Returns true if the commit threshold has been met.

    Attributes:
        tau: Probability threshold value.
    """

    def __init__(self, threshold: float) -> None:
        """Initialize the probability threshold criteria.

        Args:
            threshold: A threshold value in [0,1]. If a candidate exceeds this
                posterior probability, the system terminates.

        Raises:
            AssertionError: If threshold is not in [0,1].
        """
        assert 1 >= threshold >= 0, "stopping threshold should be in [0,1]"
        self.tau = threshold

    def decide(self, series: Dict[str, Any]) -> bool:
        """Check if probability threshold has been exceeded.

        Args:
            series: Dictionary containing series data.

        Returns:
            bool: True if threshold exceeded, False otherwise.
        """
        current_distribution = series['list_distribution'][-1]
        if np.max(current_distribution) > self.tau:
            log.info("Committing to decision: posterior exceeded threshold.")
            return True
        return False


class MarginCriteria(DecisionCriteria):
    """Criteria based on margin between top two candidates.

    This condition enforces the likelihood difference between two top
    candidates to be at least a specified value. E.g. in 4 category case with
    a margin 0.2, the edge cases [0.6,0.4,0.,0.] and [0.4,0.2,0.2,0.2]
    satisfy the condition.

    Attributes:
        margin: Required margin between top candidates.
    """

    def __init__(self, margin: float) -> None:
        """Initialize the margin criteria.

        Args:
            margin: Minimum distance required between two most likely competing
                candidates to trigger termination.

        Raises:
            AssertionError: If margin is not in [0,1].
        """
        assert 1 >= margin >= 0, "difference margin should be in [0,1]"
        self.margin = margin

    def decide(self, series: Dict[str, Any]) -> bool:
        """Check if margin between top candidates is sufficient.

        Args:
            series: Dictionary containing series data.

        Returns:
            bool: True if margin is sufficient, False otherwise.
        """
        p = copy(series['list_distribution'][-1])
        candidates = [p[idx] for idx in list(np.argsort(p)[-2:])]
        stopping_rule = np.abs(candidates[0] - candidates[1])
        d = stopping_rule > self.margin
        if d:
            log.info("Committing to decision: margin is high enough.")
        return d


class MomentumCommitCriteria(DecisionCriteria):
    """Criteria based on Shannon entropy and momentum.

    This stopping criteria combines Shannon entropy on the simplex with
    a momentum term.

    Attributes:
        lam: Linear combination parameter between entropy and speed term.
        tau: Decision threshold.
    """

    def __init__(self, tau: float, lam: float) -> None:
        """Initialize the momentum commit criteria.

        Args:
            tau: Decision threshold value.
            lam: Linear combination parameter.
        """
        self.lam = lam
        self.tau = tau

    def decide(self, series: Dict[str, Any]) -> bool:
        """Evaluate momentum-based stopping criteria.

        Args:
            series: Dictionary containing series data.

        Returns:
            bool: True if stopping criteria met, False otherwise.
        """
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

        return stopping_rule < tau_


class CriteriaEvaluator:
    """Evaluates decision criteria for BCI task control.

    This class manages multiple decision criteria to determine when to commit
    to a decision based on the accumulated evidence.

    Attributes:
        continue_criteria: List of criteria that must all be false to allow
            commitment.
        commit_criteria: List of criteria where any true value triggers
            commitment.
    """

    def __init__(self,
                 continue_criteria: Optional[List[DecisionCriteria]] = None,
                 commit_criteria: Optional[List[DecisionCriteria]] = None) -> None:
        """Initialize the criteria evaluator.

        Args:
            continue_criteria: List of criteria that must all be false to allow
                commitment.
            commit_criteria: List of criteria where any true value triggers
                commitment.
        """
        self.continue_criteria = continue_criteria or []
        self.commit_criteria = commit_criteria or []

    @classmethod
    def default(cls, min_num_inq: int, max_num_inq: int,
                threshold: float) -> 'CriteriaEvaluator':
        """Create a default CriteriaEvaluator instance.

        Args:
            min_num_inq: Minimum number of inquiries required.
            max_num_inq: Maximum number of inquiries allowed.
            threshold: Probability threshold for commitment.

        Returns:
            CriteriaEvaluator: Configured with default criteria.
        """
        return cls(
            continue_criteria=[MinIterationsCriteria(min_num_inq)],
            commit_criteria=[
                MaxIterationsCriteria(max_num_inq),
                ProbThresholdCriteria(threshold)
            ])

    def do_series(self) -> None:
        """Reset all criteria for a new series."""
        for el_ in self.continue_criteria:
            el_.reset()
        for el in self.commit_criteria:
            el.reset()

    def should_commit(self, series: Dict[str, Any]) -> bool:
        """Evaluate whether to commit to a decision.

        Args:
            series: Dictionary containing series data with:
                target (str): Target of the series.
                time_spent (np.ndarray): Time spent on each inquiry.
                list_sti (List[List[str]]): Presented symbols in each inquiry.
                list_distribution (List[np.ndarray]): Probability distributions
                    over the alphabet for each inquiry.

        Returns:
            bool: True if commitment criteria met, False otherwise.
        """
        if any(criteria.decide(series) for criteria in self.continue_criteria):
            return False
        return any(criteria.decide(series) for criteria in self.commit_criteria)
