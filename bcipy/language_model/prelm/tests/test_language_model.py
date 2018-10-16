import os
import sys
sys.path.append('.')
import unittest
from bcipy.language_model.prelm.language_model import LangModel
from errors import StatusCodeError
from bcipy.helpers.load import load_json_parameters

class TestPreLM(unittest.TestCase):

    def test_incorrect_class_variables(self):
        """
        confirm an assertion error as
        the provided fst file is invalid
        """
        abs_path_fst = os.path.abspath("fst/brown_closure.n5.kn.fst")        
        # local fst
        localfst = abs_path_fst
        # init LMWrapper
        with self.assertRaises(AssertionError):
          lmodel = LangModel(
              localfst,
              host='127.0.0.1',
              port='5000',
              logfile="lmwrap.log")

    def test_correct_process(self):
        """
        confirm the process provides the
        expected output given a correct
        input
        """
        parameters_path = 'bcipy/parameters/parameters.json'
        # call the load parameters function
        parameters = load_json_parameters(parameters_path)
        # pbsolute ath to fst
        abs_path_fst = parameters["path_to_fst"]["value"]
        # local fst
        localfst = abs_path_fst
        # init LMWrapper
        lmodel = LangModel(
            localfst,
            host='127.0.0.1',
            port='5000',
            logfile="lmwrap.log")
        # init LM
        lmodel.init()
        # get priors
        priors = lmodel.state_update(['T'])
        # display priors
        assert lmodel.decision == 'T'
        priors = lmodel.state_update(['H'])
        assert lmodel.decision == 'H'
        lmodel.reset()
        priors = lmodel.state_update(list('THE'))
        assert lmodel.decision == 'E'

#    def test_incorrect_input(self):
#        """
#        confirm the process provides
#        an error given an incorrect
#        input
#        """
#        parameters_path = 'bcipy/parameters/parameters.json'
#        # call the load parameters function
#        parameters = load_json_parameters(parameters_path)
#        # pbsolute ath to fst
#        abs_path_fst = parameters["path_to_fst"]["value"]
#        # local fst
#        localfst = abs_path_fst
#        # init LMWrapper
#        lmodel = LangModel(
#            localfst,
#            host='127.0.0.1',
#            port='5000',
#            logfile="lmwrap.log")
#        lmodel.init()
#        # try to get priors
#        with self.assertRaises(StatusCodeError) as er:
#            lmodel.state_update(['3'])

if __name__ == '__main__':
    unittest.main()
