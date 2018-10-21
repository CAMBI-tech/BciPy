import sys
sys.path.append('.')
import os
from bcipy.language_model.language_model import LangModel
from bcipy.language_model.lm_modes import lmtype
from eeg_utils import simulate_eeg

def main():
    lm = lmtype('oclm')
    # init LMWrapper
    lmodel = LangModel(lm, logfile="lmwrap.log")
    """Runs the demo"""
    # init LM
    nbest = 3
    lmodel.init(nbest)
    return_mode = 'letter'
    print("\nCharacter distribution of no history\n")
    print(lmodel.recent_priors(return_mode))
    # path to eeg samples to simulate eeg input
    path2eeg = 'bcipy/language_model/oclm/demo/EEGEvidence.txt-high'
    # eeg simulator
    simulator = simulate_eeg(path2eeg)
    
    # build evidence history
    history = "T"
    evidence = simulator.simulate(history)
    print("\nEvidence for 'T'\n")
    print(evidence)
    # feed history
    print("\nCharacter distribution of history of 'T' (what should follow 'T')\n")
    return_mode = 'letter'
    priors = lmodel.state_update(evidence, return_mode)
    # check for letter distribution
    print(lmodel.recent_priors(return_mode))
    
    # add more evidence to history
    history = "HO"
    evidence = simulator.simulate(history)
    # check for possible words
    print("\nCharacter and Word distributions of history of 'THO' (what word are we in the middle of typing)\n")
    return_mode = 'word'
    priors = lmodel.state_update(evidence, return_mode)
    print(lmodel.recent_priors(return_mode))
    lmodel.reset()
    print("\n--------------RESET-------------\n")
    
    # build evidence history
    history = "YO"
    evidence = simulator.simulate(history)
    print("\nEvidence for 'YO'\n")  # a likelihood domain (the higher the more likely)
    print(evidence)
    return_mode = 'word'
    priors = lmodel.state_update(evidence, return_mode)
    print("\nCharacter and Word distributions of 'YO' to follow it, and word to autocomplete\n")
    print(priors) # a negative likelihood domain (the lower the more likely)

if __name__ == "__main__":
    main()
