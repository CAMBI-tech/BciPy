import numpy as np
from acquisition.sig_pro.sig_pro import sig_pro

def inference(x, y, model, alphabet):
    """ Evaluates the log likelihood ratios given the model and input.
        Then maps the results in the alphabet.
        Args:
            x(ndarray[float]): raw input sequence c x N
            y(ndarray[int]): N x 1 label array
            model(pipeline): trained likelihood model
            alphabet: 
        Return:
            scores(ndarray): N x c  log-likelihood array
                    N is number of samples c is number of classes
            """
    # This returns a sequence that is filtered and downsampled
    dat = sig_pro(input_seq = x)
    # This evaluates the loglikelihood probabilities for p(e|l=1) and p(e|l=0)
    scores = np.exp(model.transform(dat))
    # This evaluates the log likelihood ratios:
    scores = -scores[:,1] / scores[:,0]
    # This maps the log likelihood ratios to alphabet:
    # If the letter in alphabet does not exist in y, it takes 1
    # the likelihood ratio.
    lik_r = np.ones(len(alphabet))
    for i in range(len(alphabet)):
        for j in range(len(y)):
            if alphabet(i) == y(j):
                lik_r[i] = scores[j]
    return lik_r
