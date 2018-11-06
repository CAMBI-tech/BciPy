import sys
sys.path.append('.')
from bcipy.language_model.prelm_language_model import LangModel
from bcipy.helpers.lang_model_related import norm_domain


def main():
    """Runs the demo"""
    # init LMWrapper
    lmodel = LangModel(logfile="lmwrap.log")
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
    print(lmodel.recent_priors())
    # reset history al together
    lmodel.reset()
    print("\n--------------RESET-------------\n")
    print('\nHistory: THE (fed as a single string)\n')
    priors = lmodel.state_update(list('THE'))
    print(lmodel.recent_priors())

    print('\n\nLetters in the probability domain:\n')
    print(norm_domain(lmodel.recent_priors()['letter']))


if __name__ == "__main__":
    main()
