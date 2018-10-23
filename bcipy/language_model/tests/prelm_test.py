import os
import sys
sys.path.append('.')
import unittest
from bcipy.language_model.lm_modes import LmType, LangModel
from bcipy.language_model.errors import StatusCodeError

class TestPreLM(unittest.TestCase):

    def test_correct_process(self):
        """
        confirm the process provides the
        expected prior output given a correct
        input
        """
        lm = LmType.PRELM
        # init LMWrapper
        lmodel = LangModel(lm, logfile="lmwrap.log")
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

#    def test_incorrect_input(self):
#        """
#        confirm the process provides
#        an error given an incorrect
#        input
#        """
#        lm = LmType.PRELM
#        # init LMWrapper
#        lmodel = LangModel(lm, logfile="lmwrap.log")
#        lmodel.init()
#        # try to get priors
#        with self.assertRaises(StatusCodeError) as er:
#            lmodel.state_update(['3'])

if __name__ == '__main__':
    unittest.main()
