import numpy as np


x = np.ones((16,256*10))












def sigPro(inputSeq, filt = None, fs = 256, k = 2):
    # import filters from database file

    f = open('filterDict.txt', 'r')

    filterDict = eval(f.read());




    if filt != None:
        filt = filterDict(fs);



    outputSeq = 1;
    return outputSeq