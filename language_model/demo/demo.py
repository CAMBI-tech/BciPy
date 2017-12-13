from bci.language_model import LangModel

# local fst
localfst = "/Users/dudy/CSLU/bci/5th_year/letters/pywrapper/lm/brown_closure.n5.kn.fst"
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
lmodel.recent_priors()
priors = lmodel.state_update(['H'])
lmodel.recent_priors()
priors = lmodel.state_update(['E'])
# reset history al together
lmodel.reset()
lmodel.recent_priors()
priors = lmodel.state_update(list('THE'))
lmodel.recent_priors()
