from LMWrapper import LangModel

"""
This script shows how to use the language model wrapper
which is encapsulated in LMWrapper module
"""

# local fst
localfst = "the path to the fst file"
# init LMWrapper
lmodel = LangModel(
    localfst,
    host='127.0.0.1',
    port='5000',
    logfile="lmwrap.log")
# init LM
lmodel.init()
# get priors
priors = lmodel.state_update('t')
# display priors
lmodel.recent_priors()
priors = lmodel.state_update('h')
lmodel.recent_priors()
priors = lmodel.state_update('e')
# reset history al together
lmodel.reset()
lmodel.recent_priors()
priors = lmodel.state_update(list('the'))
lmodel.recent_priors()
