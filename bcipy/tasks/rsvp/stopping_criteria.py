import numpy as np
import logging
from typing import Dict, List

log = logging.getLogger(__name__)


# Criteria
class DecisionCriteria():
    """Abstract class for Criteria which can be applied to evaluate a sequence
    """

    def apply(self, epoch, commit_params):
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
            commit_params - params relevant to stoppage criteria
                min_num_seq: int - minimum number of sequences required
                max_num_seq: int - max number of sequences allowed
                threshold: float - minimum likelihood required
        """
        raise NotImplementedError()


class MinIterationsCriteria(DecisionCriteria):
    """Returns true if the minimum number of iterations have not yet been reached."""

    def apply(self, epoch, commit_params):
        # Note: we use 'list_sti' parameter since this is the number of
        # sequences displayed. The length of 'list_distribution' is 1 greater
        # than this, since the language model distribution is added before
        # the first sequence is displayed.
        current_seq = len(epoch['list_sti'])
        log.debug(
            f"Checking min iterations; current iteration is {current_seq}")
        return current_seq < commit_params['min_num_seq']


class DecreasedProbabilityCriteria(DecisionCriteria):
    """Returns true if the letter with the max probability decreased from the
        last sequence."""

    def apply(self, epoch, commit_params):
        if len(epoch['list_distribution']) < 2:
            return False
        prev_dist = epoch['list_distribution'][-2]
        cur_dist = epoch['list_distribution'][-1]
        return np.argmax(cur_dist) == np.argmax(
            prev_dist) and np.max(cur_dist) < np.max(prev_dist)


class MaxIterationsCriteria(DecisionCriteria):
    """Returns true if the max iterations have been reached."""

    def apply(self, epoch, commit_params):
        # Note: len(epoch['list_sti']) != len(epoch['list_distribution'])
        # see MinIterationsCriteria comment
        current_seq = len(epoch['list_sti'])
        if current_seq >= commit_params['max_num_seq']:
            log.debug(
                "Committing to decision: max iterations have been reached.")
            return True
        return False


class CommitThresholdCriteria(DecisionCriteria):
    """Returns true if the commit threshold has been met."""

    def apply(self, epoch, commit_params):
        current_distribution = epoch['list_distribution'][-1]
        if np.max(current_distribution) > commit_params['threshold']:
            log.debug("Committing to decision: Likelihood exceeded threshold.")
            return True
        return False


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
    def default(cls):
        return cls(continue_criteria=[MinIterationsCriteria()],
                   commit_criteria=[
                       MaxIterationsCriteria(),
                       CommitThresholdCriteria()
                   ])

    def should_commit(self, epoch: Dict, params: Dict):
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
            params - params relevant to stoppage criteria
                min_num_seq: int - minimum number of sequences required
                max_num_seq: int - max number of sequences allowed
                threshold: float - minimum likelihood required
        """
        if any(
                criteria.apply(epoch, params)
                for criteria in self.continue_criteria):
            return False
        return any(
            criteria.apply(epoch, params) for criteria in self.commit_criteria)
