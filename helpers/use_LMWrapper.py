from lm_wrapper import LangModel

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
