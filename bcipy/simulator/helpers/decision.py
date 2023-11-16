from abc import ABC

import numpy as np
import logging
from typing import Dict, List
from copy import copy

from bcipy.simulator.helpers.types import InquiryResult

log = logging.getLogger(__name__)


# TODO copied over criteria.py. Still debating whether to tie our simulator to that class or not. For now, just hotfixing logic to fit with my implementation

class SimDecisionCriteria(ABC):
    """Abstract class for Criteria which can be applied to evaluate a inquiry
    """

    def reset(self):
        pass

    def decide(self, series: List[InquiryResult]):
        # TODO update documentation
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


class MinIterations(SimDecisionCriteria):
    """ Returns true if the minimum number of iterations have not yet
        been reached. """

    def __init__(self, min_num_inq: int):
        """ Args:
                min_num_inq(int): minimum number of inquiry number before any
                 termination objective is allowed to be triggered """
        self.min_num_inq = min_num_inq

    def decide(self, series: List[InquiryResult]):
        # Note: we use 'list_sti' parameter since this is the number of
        # inquiries displayed. The length of 'list_distribution' is 1 greater
        # than this, since the language model distribution is added before
        # the first inquiry is displayed.
        current_inq = len(series) if series else 0
        log.debug(
            f"Checking min iterations; current iteration is {current_inq}")
        return current_inq < self.min_num_inq


#
# class DecreasedProbabilityCriteria(DecisionCriteria):
#     """Returns true if the letter with the max probability decreased from the
#         last inquiry."""
#
#     def decide(self, series: Dict):
#         if len(series['list_distribution']) < 2:
#             return False
#         prev_dist = series['list_distribution'][-2]
#         cur_dist = series['list_distribution'][-1]
#         return np.argmax(cur_dist) == np.argmax(
#             prev_dist) and np.max(cur_dist) < np.max(prev_dist)
#
#
class MaxIterationsSim(SimDecisionCriteria):
    """Returns true if the max iterations have been reached."""

    def __init__(self, max_num_inq: int):
        """ Args:
                max_num_inq(int): maximum number of inquiries allowed before
                    mandatory termination """
        self.max_num_inq = max_num_inq

    def decide(self, series: List[InquiryResult]):
        # Note: len(series['list_sti']) != len(series['list_distribution'])
        # see MinIterationsCriteria comment
        current_inq = len(series) if series else 0
        if current_inq >= self.max_num_inq:
            log.debug(
                "Committing to decision: max iterations have been reached.")
            return True
        return False


class ProbThresholdSim(SimDecisionCriteria):
    """Returns true if the commit threshold has been met."""

    def __init__(self, threshold: float):
        """ Args:
                threshold(float in [0,1]): A threshold on most likely
                    candidate posterior. If a candidate exceeds a posterior
                    the system terminates.
                 """
        assert 1 >= threshold >= 0, "stopping threshold should be in [0,1]"
        self.tau = threshold

    def decide(self, series: List[InquiryResult]):
        current_distribution = series[-1].fused_likelihood
        if np.max(current_distribution) > self.tau:
            log.debug("Committing to decision: posterior exceeded threshold.")
            return True
        return False