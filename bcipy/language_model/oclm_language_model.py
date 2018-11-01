import logging
import sys
import math
from typing import List
from collections import defaultdict
from bcipy.helpers.bci_task_related import alphabet
from bcipy.language_model import lm_server
from bcipy.language_model.errors import (EvidenceDataStructError,
                                         NBestError,
                                         NBestHighValue)
from bcipy.language_model.lm_server import LmServerConfig

sys.path.append('.')
ALPHABET = alphabet()
LM_SPACE = '#'

class LangModel:

    DEFAULT_CONFIG = LmServerConfig(
        image="oclmimage:version2.0",
        port=6000,
        docker_port=5000)

    def __init__(self, server_config: LmServerConfig=DEFAULT_CONFIG,
                 logfile: str = "log"):
        """
        Initiate the langModel class and starts the corresponding docker
        server for the given type.

        Input:
          lmtype - language model type
          logfile - a valid filename to function as a logger
        """
        self.server_config = server_config
        self.priors = defaultdict(list)
        logging.basicConfig(filename=logfile, level=logging.INFO)
        lm_server.start(server_config)

    def init(self, nbest: int = 1):
        """
        Initialize the language model (on the server side)
        Input:
            nbest - top N symbols from evidence
        """
        if not isinstance(nbest, int):
            raise NBestError(nbest)
        if nbest > 4:
            raise NBestHighValue(nbest)
        lm_server.post_json_request(
            self.server_config, 'init', data={'nbest': nbest})

    def cleanup(self):
        """Stop the docker server"""
        lm_server.stop(self.server_config)

    def reset(self):
        """
        Clean observations of the language model use reset
        """
        lm_server.post_json_request(self.server_config, 'reset')
        self.priors = defaultdict(list)
        logging.info("\ncleaning history\n")

    def state_update(self, evidence: List, return_mode: str = 'letter'):
        """
        Provide a prior distribution of the language model
        in return to the system's decision regarding the
        last observation
        Both lm types allow providing more the one timestep
        input. Pay attention to the data struct expected.
        OCLM
        Input:
            evidence - a list of (list of) tuples [[(sym1, prob), (sym2, prob2)]]
            the numbers are assumed to be in the log probabilty domain
            return_mode - 'letter' or 'word' (available
                          for oclm) strings
        Output:
            priors - a json dictionary with Normalized priors
                     in the Negative Log probabilty domain.
        """

        # assert the input contains a valid symbol
        try:
            clean_evidence = []
            for tmp_evidence in evidence:
                tmp = []
                for (symbol, pr) in tmp_evidence:
                    assert symbol in ALPHABET, \
                        "%r contains invalid symbol" % evidence
                    if symbol == "_":
                        tmp.append(("#", pr))
                    else:
                        tmp.append((symbol.lower(), pr))
                clean_evidence.append(tmp)
        except:
            raise EvidenceDataStructError

        output = lm_server.post_json_request(self.server_config,
                                             'state_update',
                                             {'evidence': clean_evidence,
                                              'return_mode': return_mode})
        return self.__return_priors(output, return_mode)

    def _logger(self):
        """
        Log the priors given the recent decision
        """
        # print a json dict of the priors
        logging.info('\nThe priors are:\n')
        for k in self.priors.keys():
            priors = self.priors[k]
            logging.info('\nThe priors for {0} type are:\n'.format(k))
            for (symbol, pr) in priors:
                logging.info('{0} {1:.4f}'.format(symbol, pr))

    def recent_priors(self, return_mode='letter'):
        """
        Display the priors given the recent decision
        """

        if not bool(self.priors[return_mode]):
            output = lm_server.post_json_request(self.server_config,
                                                 'recent_priors',
                                                 {'return_mode': return_mode})
            return self.__return_priors(output, return_mode)
        else:
            return self.priors

    def __return_priors(self, output, return_mode):
        """
        A helper function to provide the desired output
        depending on the return_mode
        """

        self.priors = defaultdict(list)
        self.priors['letter'] = [
            (letter.upper(), prob)
            if letter != LM_SPACE
            else ("_", prob)
            for (letter, prob) in output['letter']]

        if return_mode != 'letter':
            self.priors[return_mode] = list(map(tuple, output[return_mode]))
        return self.priors
