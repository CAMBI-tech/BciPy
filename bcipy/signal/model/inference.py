import numpy as np


def inference(x, targets, model, alphabet):
    """Evaluate the log likelihood ratios given the model and input.

    Then maps the distribution over the alphabet.
    Args:
        x(ndarray(float)): 3 dimensional np array first dimension is
            channels second dimension is trials and third dimension is
            time samples.
        targets(ndarray[str]): flashed symbols in order.
        model(pipeline): trained likelihood model.
        alphabet(list[str]): symbol set. Letters in the alphabet. All uppercase
    Return:
        lik_r(ndarray[float]): likelihood array.
    """

    # Evaluates the likelihood probabilities for p(e|l=1) and p(e|l=0)
    scores = np.exp(model.transform(x))
    # Evaluates the likelihood ratios
    scores = scores[:, 1] / (scores[:, 0] + np.power(0.1, 10)) + np.power(0.1, 10)
    # print("These values should be between 0 and 1: ", scores)

    # This maps the likelihood distribution over the alphabet
    #   If the letter in the alphabet does not exist in the target string,
    #       it takes 1
    lik_r = np.ones(len(alphabet))
    for idx in range(len(scores)):
        lik_r[alphabet.index(targets[idx])] *= scores[idx]

    return lik_r
