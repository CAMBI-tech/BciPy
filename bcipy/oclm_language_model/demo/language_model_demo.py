import sys
sys.path.insert(0, ".")
import os
from bcipy.oclm_language_model.language_model import LangModel
from eeg_utils import simulate_eeg

# init LMWrapper
lmodel = LangModel(
    host='127.0.0.1',
    port='5000',
    logfile="lmwrap.log")

# init LM
nbest = 3
lmodel.init(nbest)

# path to eeg samples to simulate eeg input
path2eeg = 'bcipy/oclm_language_model/demo/EEGEvidence.txt-high'
# eeg simulator
simulator = simulate_eeg(path2eeg)

# build evidence history
history = "T"
evidence = simulator.simulate(history)
# feed history
print("\nCharacter distribution of no history\n")
return_mode = 'letter'
priors = lmodel.state_update(evidence, return_mode)
# check for letter distribution
print(priors)
# print lmodel.recent_priors()

print("\nCharacter and Word distributions of history of 'TH'\n")
# add more evidence to history
history = "HO"
evidence = simulator.simulate(history)

# check for possible words
return_mode = 'word'
priors = lmodel.state_update(evidence, return_mode)
print(lmodel.recent_priors())
lmodel.reset()

print("\nCharacter and Word distributions of 'Y'\n")
# build evidence history
history = "YO"
evidence = simulator.simulate(history)
print("Evidence")  # a likelihood domain (the higher the more likely)
print(evidence)
return_mode = 'word'
priors = lmodel.state_update(evidence, return_mode)
print("Priors")  # a negative likelihood domain (the lower the more likely)
print(lmodel.recent_priors())
