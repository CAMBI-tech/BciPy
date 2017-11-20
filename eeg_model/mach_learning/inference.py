import numpy as np
from acquisition.sig_pro.sig_pro import sig_pro
from eeg_model.mach_learning.trial_reshaper import trial_reshaper

def inference(x, y, model, alphabet, trigger_loc):
    """ 
        Evaluates the log likelihood ratios given the model and input.
        Then maps the results in the alphabet.
        Args:
            x(ndarray[float]): Input sequence to be filtered. Expected dimensions are 16xT
            y(ndarray[int]): Label array
            model(pipeline): Trained likelihood model
            trigger_loc: Location of the trigger.txt file to read triggers from
            alphabet: 
        Return:
            lik_r(ndarray): Log-likelihood ratio array   
    """

    # This returns a sequence that is filtered and downsampled
    filtered_eeg = sig_pro(input_seq = x)
    # Reshape the data:
    labels = trial_reshaper(trigger_loc, filtered_eeg, fs = 256, k = 2)
    # This evaluates the loglikelihood probabilities for p(e|l=1) and p(e|l=0)
    scores = np.exp(model.transform(labels))
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
