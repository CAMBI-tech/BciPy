import logging
import sys
import math
from typing import List
from collections import defaultdict
from bcipy.helpers.bci_task_related import alphabet
from bcipy.language_model import lm_server
from bcipy.language_model.lm_server import LmServerConfig

sys.path.append('.')
ALPHABET = alphabet()


class LangModel:

    def __init__(self, server_config: LmServerConfig, logfile: str = "log"):
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
        lm_server.start(self.server_config)

    def init(self, domain: str = 'log', nbest: int = 1):
        """
        Initialize the language model (on the server side)
        Input:
            nbest - top N symbols from evidence
            domain - a string to indicate to domain
                     of expected output. 
                    'log' is of the negative log domain
                    'norm' is of the probabilty domain
        """
        assert isinstance(domain, str)
        self.domain = domain
        lm_server.post_json_request(
            self.server_config, 'init', data={'nbest': nbest})

    def reset(self):
        """
        Clean observations of the language model use reset
        """
        lm_server.post_json_request(self.server_config, 'reset')
        logging.info("\ncleaning history\n")

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
            the numbers are assumed to be in the log probabilty domain
            return_mode - 'letter' or 'word' (available
                          for oclm) strings
        Output:
            priors - a json dictionary with Normalized priors
                     in the Negative Log probabilty domain
                     (the default is log) or the probability
                     domain
        """
        assert return_mode == 'letter', "PRELM only allows letter output"
        # assert the input contains a valid symbol
        decision = evidence  # in prelm the we treat it as a decision
        for symbol in decision:
            assert symbol in ALPHABET or ' ', \
                "%r contains invalid symbol" % decision
        clean_evidence = []
        for symbol in decision:
            if symbol == '_':
                symbol = '#'
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
        depending on the return_mode and the domain
        of probaiblities requested
        """

        self.priors = defaultdict(list)
        if self.domain == 'log':
            self.priors['letter'] = [
                [letter.upper(), prob]
                if letter != '#' # hard coded (to deal with in future)
                else ["_", prob]
                for (letter, prob) in output['letter']]

        else:
            self.priors['letter'] = [
                [letter.upper(), math.e**(-prob)]
                if letter != '#'
                else ["_", math.e**(-prob)]
                for (letter, prob) in output['letter']]

        return self.priors
