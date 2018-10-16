import sys

from bcipy.helpers.load import load_json_parameters
from bcipy.language_model.language_model import LangModel

sys.path.append('.')


def main():
    """Runs the demo"""
    parameters_path = 'bcipy/parameters/parameters.json'
    # call the load parameters function
    parameters = load_json_parameters(parameters_path)
    # pbsolute ath to fst
    abs_path_fst = parameters["path_to_fst"]["value"]
    print(abs_path_fst)
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
    print(lmodel.recent_priors())
    priors = lmodel.state_update(['H'])
    print(lmodel.recent_priors())
    priors = lmodel.state_update(['E'])
    # reset history al together
    lmodel.reset()
    print(lmodel.recent_priors())
    priors = lmodel.state_update(list('THE'))
    print(lmodel.recent_priors())


if __name__ == "__main__":
    main()
