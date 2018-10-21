class lmtype:
    def __init__(self, lmtype):

       self.type = lmtype
       self.host = "127.0.0.1"
       self.dport = "5000"

       if lmtype == 'oclm':
           self.port = "6000"
           self.image = "oclmimage:version2.0"
           self.nbest = "1"
           
       elif lmtype == 'prelm':
           self.port = "5000"
           self.image = "lmimage:version2.0"
           self.localfst = "/Users/dudy/CSLU/bci/BciPy/bcipy/language_model/fst/brown_closure.n5.kn.fst"

