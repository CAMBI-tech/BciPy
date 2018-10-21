import sys
sys.path.append('.')
from bcipy.language_model.language_model import LangModel
from bcipy.language_model.lm_modes import lmtype

def main():
    lm = lmtype('prelm')
    # init LMWrapper
    lmodel = LangModel(lm, logfile="lmwrap.log")
    """Runs the demo"""
    # init LM
    lmodel.init()
    print('\nNo History\n')
    # get initial priors
    print(lmodel.recent_priors())
    # get priors
    print('\nHistory: T\n')
    priors = lmodel.state_update(['T'])
    # display priors
    print(lmodel.recent_priors())
    print('\nHistory: TH\n')
    priors = lmodel.state_update(['H'])
    print(lmodel.recent_priors())
    print('\nHistory: THE\n')
    priors = lmodel.state_update(['E'])
    # reset history al together
    lmodel.reset()
    print(lmodel.recent_priors())
    print('\nHistory: THE (fed as a single string)\n')
    priors = lmodel.state_update(list('THE'))
    print(lmodel.recent_priors())


if __name__ == "__main__":
    main()
