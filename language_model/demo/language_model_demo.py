import sys
sys.path.insert(0, ".")
import os
from language_model.language_model import LangModel

# path to fst in bci repo on local machiene
abs_path_fst = os.path.abspath("./language_model/fst/brown_closure.n5.kn.fst")

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
print lmodel.recent_priors()
priors = lmodel.state_update(['H'])
print lmodel.recent_priors()
priors = lmodel.state_update(['E'])
# reset history al together
lmodel.reset()
print lmodel.recent_priors()
priors = lmodel.state_update(list('THE'))
print lmodel.recent_priors()
