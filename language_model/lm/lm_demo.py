from LMWrapper import LangModel

"""
This script shows how to use the language model wrapper
which is encapsulated in LMWrapper module
"""
def init_language_model(localfst, host, port, logfile):  
  lmodel = LangModel(localfst, host, port, logfile)
  return lmodel
# local fst
localfst = "the path to the fst file"
# init LMWrapper
lmodel = init_language_model(localfst, host='127.0.0.1', port='5000', logfile="lmwrap.log")
# init LM
lmodel.init()
# get priors
priors = lmodel.state_update('T')
# display priors
lmodel.recent_priors()
priors = lmodel.state_update('H')
lmodel.recent_priors()
priors = lmodel.state_update('E')
# reset history al together
lmodel.reset()
lmodel.recent_priors()
priors = lmodel.state_update(list('THE'))
lmodel.recent_priors()
