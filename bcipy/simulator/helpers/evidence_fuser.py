from abc import abstractmethod, ABC
from typing import Optional, Dict

import numpy as np

from bcipy.simulator.helpers.types import SimEvidence


class EvidenceFuser(ABC):

    @abstractmethod
    def fuse(self, prior_likelihood: Optional[np.ndarray],
             evidence: Dict[str, SimEvidence]) -> np.ndarray:
        ...

    @staticmethod
    def make_prior(len_dist):
        return np.ones(len_dist) / len_dist


class MultiplyFuser(EvidenceFuser):

    def __init__(self):
        pass

    def fuse(self, prior_likelihood, evidence) -> np.ndarray:

        distribution_shape = len(list(evidence.values())[0].evidence)
        prior_likelihood = prior_likelihood if prior_likelihood is not None \
            else EvidenceFuser.make_prior(distribution_shape)

        ret_likelihood = prior_likelihood.copy()

        for sim_evidence in evidence.values():
            likelihoods = sim_evidence.evidence
            assert ret_likelihood.shape == likelihoods.shape
            ret_likelihood *= likelihoods[:]
        ret_likelihood = self.__normalize_likelihood(ret_likelihood)

        return ret_likelihood

    def __normalize_likelihood(self, likelihood):

        cleaned_likelihood = likelihood.copy()
        if np.isinf(np.sum(likelihood)):
            tmp = np.zeros(len(likelihood))
            tmp[np.where(likelihood == np.inf)[0][0]] = 1
            cleaned_likelihood = tmp

        if not np.isnan(np.sum(likelihood)):
            cleaned_likelihood = likelihood / np.sum(likelihood)

        return cleaned_likelihood
