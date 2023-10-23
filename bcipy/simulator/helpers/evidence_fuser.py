from abc import abstractmethod, ABC
from typing import Optional, Dict

import numpy as np


class EvidenceFuser(ABC):

    @abstractmethod
    def fuse(self, prior_likelihood: Optional[np.ndarray], evidence: Dict) -> np.ndarray:
        ...


class MultipyFuser(EvidenceFuser):

    def __init__(self):
        pass

    def fuse(self, prior_likelihood, evidence) -> np.ndarray:

        len_dist = len(list(evidence.values())[0])
        prior_likelihood = prior_likelihood if prior_likelihood is not None else self.__make_prior(len_dist)
        ret_likelihood = prior_likelihood.copy()

        for value in evidence.values():
            ret_likelihood *= value[:]
        ret_likelihood = self.__clean_likelihood(ret_likelihood)

        return ret_likelihood

    def __make_prior(self, len_dist):
        return np.ones(len_dist) / len_dist

    def __clean_likelihood(self, likelihood):

        cleaned_likelihood = likelihood.copy()
        if np.isinf(np.sum(likelihood)):
            tmp = np.zeros(len(likelihood))
            tmp[np.where(likelihood == np.inf)[0][0]] = 1
            cleaned_likelihood = tmp

        if not np.isnan(np.sum(likelihood)):
            cleaned_likelihood = likelihood / np.sum(likelihood)

        return cleaned_likelihood
