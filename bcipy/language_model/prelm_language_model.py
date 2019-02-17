import logging
import sys
import math
from typing import List
from collections import defaultdict
from bcipy.helpers.bci_task_related import alphabet, SPACE_CHAR
from bcipy.language_model import lm_server
from bcipy.language_model.lm_server import LmServerConfig
from bcipy.helpers.system_utils import dot
log = logging.getLogger(__name__)
sys.path.append('.')
ALPHABET = alphabet()
LM_SPACE = '#'


class LangModel:
    DEFAULT_CONFIG = LmServerConfig(
        image="lmimage:version2.0",
        port=5000,
        docker_port=5000,
        volumes={dot(__file__, 'fst', 'brown_closure.n5.kn.fst'):
                 "/opt/lm/brown_closure.n5.kn.fst"})

    def __init__(self, server_config: LmServerConfig = DEFAULT_CONFIG,
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
        log.setLevel(logging.INFO)
        log.addHandler(logging.FileHandler(logfile))
        lm_server.start(self.server_config)

    def init(self, nbest: int = 1):
        """
        Initialize the language model (on the server side)
        Input:
            nbest - top N symbols from evidence
        """
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
        log.info("\ncleaning history\n")

    def state_update(self, evidence: List, return_mode: str = 'letter'):
        """
        Provide a prior distribution of the language model
        in return to the system's decision regarding the
        last observation
        Both lm types allow providing more the one timestep
        input. Pay attention to the data struct expected.

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
        assert return_mode == 'letter', "PRELM only allows letter output"
        # assert the input contains a valid symbol
        decision = evidence  # in prelm the we treat it as a decision
        for symbol in decision:
            assert symbol in ALPHABET or ' ', \
                "%r contains invalid symbol" % decision
        clean_evidence = []
        for symbol in decision:
            if symbol == SPACE_CHAR:
                symbol = LM_SPACE
            clean_evidence.append(symbol.lower())

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
        log.info('\nThe priors are:\n')
        for k in self.priors.keys():
            priors = self.priors[k]
            log.info('\nThe priors for {0} type are:\n'.format(k))
            for (symbol, pr) in priors:
                log.info('{0} {1:.4f}'.format(symbol, pr))

    def recent_priors(self, return_mode='letter'):
        """
        Display the priors given the recent decision
        """
        assert return_mode == 'letter', "PRELM only allows letter output"
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
        depending on the return_mode.
        """

        self.priors = defaultdict(list)
        self.priors['letter'] = [
            (letter.upper(), prob)
            if letter != LM_SPACE  # hard coded (to deal with in future)
            else ("_", prob)
            for (letter, prob) in output['letter']]
        return self.priors
