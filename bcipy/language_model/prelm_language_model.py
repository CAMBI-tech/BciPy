"""Defines the PRELM language model"""
import logging
import sys
from typing import List, Tuple, Union, Optional
from collections import defaultdict
from bcipy.helpers.task import SPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.language_model import lm_server
from bcipy.language_model.lm_server import LmServerConfig
from bcipy.helpers.system_utils import dot

log = logging.getLogger(__name__)
sys.path.append('.')
LM_SPACE = '#'


class PrelmLanguageModel(LanguageModel):
    """PRELM character language model."""
    DEFAULT_CONFIG = LmServerConfig(
        image="lmimage:version2.0",
        port=5000,
        docker_port=5000,
        volumes={
            dot(__file__, 'fst', 'brown_closure.n5.kn.fst'):
            "/opt/lm/brown_closure.n5.kn.fst"
        })

    def __init__(self,
                 response_type: Optional[ResponseType] = None,
                 symbol_set: Optional[List[str]] = None,
                 lang_model_server_port: int = 5000):
        """
        Initiate the langModel class and starts the corresponding docker
        server for the given type.

        Input:
          lmtype - language model type
          logfile - a valid filename to function as a logger
        """
        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.server_config = PrelmLanguageModel.DEFAULT_CONFIG
        self.server_config.port = lang_model_server_port

        self.priors = defaultdict(list)
        log.setLevel(logging.INFO)
        log.addHandler(logging.FileHandler("lmwrap.log"))
        lm_server.start(self.server_config)
        self.init()

    def supported_response_types(self):
        return [ResponseType.SYMBOL]

    def init(self, nbest: int = 1):
        """
        Initialize the language model (on the server side)
        Input:
            nbest - top N symbols from evidence
        """
        lm_server.post_json_request(self.server_config,
                                    'init',
                                    data={'nbest': nbest})

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

    def predict(self, evidence: Union[str, List[str]]) -> List[Tuple]:
        """Performs `state_update` and subsequently resets the model."""
        priors = self.state_update(evidence=evidence)
        self.reset()
        return priors['letter']

    def update(self) -> None:
        """Update the model state"""

    def load(self) -> None:
        """Restore model state from the provided checkpoint"""

    def state_update(self, evidence: Union[str, List[str]]):
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
        Output:
            priors - a json dictionary with Normalized priors
                     in the Negative Log probability domain.
        """

        # assert the input contains a valid symbol
        decision = evidence  # in prelm the we treat it as a decision
        for symbol in decision:
            assert symbol in self.symbol_set or ' ', \
                "%r contains invalid symbol" % decision
        clean_evidence = []
        for symbol in decision:
            if symbol == SPACE_CHAR:
                symbol = LM_SPACE
            clean_evidence.append(symbol.lower())

        return_mode = 'letter'
        output = lm_server.post_json_request(self.server_config,
                                             'state_update', {
                                                 'evidence': clean_evidence,
                                                 'return_mode': return_mode
                                             })

        return self.__return_priors(output)

    def _logger(self):
        """
        Log the priors given the recent decision
        """
        # print a json dict of the priors
        log.info('\nThe priors are:\n')
        for k in self.priors.keys():
            priors = self.priors[k]
            log.info('\nThe priors for {0} type are:\n'.format(k))
            for (symbol, prior) in priors:
                log.info('{0} {1:.4f}'.format(symbol, prior))

    def recent_priors(self, return_mode='letter'):
        """
        Display the priors given the recent decision
        """
        assert return_mode == 'letter', "PRELM only allows letter output"
        if not bool(self.priors[return_mode]):
            output = lm_server.post_json_request(self.server_config,
                                                 'recent_priors',
                                                 {'return_mode': return_mode})
            return self.__return_priors(output)

        return self.priors

    def __return_priors(self, output):
        """
        A helper function to provide the desired output
        depending on the return_mode.
        """

        self.priors = defaultdict(list)
        self.priors['letter'] = [
            (letter.upper(),
             prob) if letter != LM_SPACE  # hard coded (to deal with in future)
            else ("_", prob) for (letter, prob) in output['letter']
        ]
        return self.priors
