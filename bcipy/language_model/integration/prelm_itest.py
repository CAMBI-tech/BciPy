import unittest

from bcipy.language_model.prelm_language_model import PrelmLanguageModel
from bcipy.language_model.errors import StatusCodeError


class TestPreLM(unittest.TestCase):

    def test_correct_process(self):
        """
        confirm the process provides the
        expected prior output given a correct
        input
        """
        # init LMWrapper
        lmodel = PrelmLanguageModel()
        # init LM
        lmodel.init()
        # get priors
        priors = lmodel.state_update(['T'])
        # display priors
        assert priors['letter'][0][0] == 'H'
        priors = lmodel.state_update(['H'])
        assert priors['letter'][0][0] == 'E'
        lmodel.reset()
        priors = lmodel.state_update(list('THE'))
        assert priors['letter'][0][0] == '_'

    def test_incorrect_input(self):
        """
        confirm the process provides
        an error given an incorrect
        input
        """
        # init LMWrapper
        lmodel = PrelmLanguageModel()
        lmodel.init()
        # try to get priors
        with self.assertRaises(StatusCodeError):
            lmodel.state_update(['3'])


if __name__ == '__main__':
    unittest.main()
