# pylint: disable=unused-argument
"""Language model that always gives equal priority to every letter."""
import math
from itertools import repeat
from typing import List
from collections import defaultdict
from bcipy.helpers.bci_task_related import alphabet
from bcipy.language_model.lm_server import LmServerConfig

ALPHABET = alphabet()


class LangModel:
    """Language model that always gives equal priority to every letter.
    Represents the default behavior if a language model is not enabled."""

    DEFAULT_CONFIG = LmServerConfig(None)

    def __init__(self, server_config: LmServerConfig = DEFAULT_CONFIG,
                 logfile: str = "log"):
        """
        Initiate the langModel class and starts the corresponding docker
        server for the given type.

        Input:
          lmtype - language model type
          logfile - a valid filename to function as a logger
        """

        self.priors = defaultdict(list)
        self.priors['letter'] = list(
            zip(ALPHABET, repeat(-math.log(1 / len(ALPHABET)))))

    def init(self, nbest: int = 1):
        """
        Initialize the language model (on the server side)
        Input:
            nbest - top N symbols from evidence
        """
        pass

    def cleanup(self):
        """Stop the docker server"""
        pass

    def reset(self):
        """
        Clean observations of the language model use reset
        """
        pass

    def state_update(self, evidence: List, return_mode: str = 'letter'):
        """
        Provide a prior distribution of the language model
        in return to the system's decision regarding the
        last observation

        Input:
            decision - a symbol or a string of symbols in encapsulated in a
            list
            the numbers are assumed to be in the log probability domain
            return_mode - 'letter' or 'word' (available
                          for oclm) strings
        Output:
            priors - a json dictionary with Normalized priors
                     in the Negative Log probability domain.
        """
        assert return_mode == 'letter'
        return self.priors

    def recent_priors(self, return_mode='letter'):
        """
        Display the priors given the recent decision
        """
        assert return_mode == 'letter'
        return self.priors
