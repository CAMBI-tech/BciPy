import torch
from torch import Tensor


def update_log_posterior(
    model_log_probs: Tensor,
    log_prior: Tensor,
    presented_seq_idx: Tensor,
    none_class_idx: int,
    alpha: float = None,
):
    """
    Update estimated posterior over the alphabet.
    Each item in the batch updates a separate posterior.
    (TODO - giving a batch should represent several updates to make to a single posterior)

    Args:
        model_log_probs (Tensor): predicted class log probabilities from the pretrained model
        log_prior (Tensor): shape (batch_size, alphabet_len): prob vector over alphabet, ordered
        presented_seq_idx -(Tensor): shape (batch_size, seq_len), indices of letters used, in order of appearance
        none_class_idx: 0 or seq_len - the index of the "none" class from classifier's output.
                Notice this is either the first or last index of the model's (seq_len+1) output values.
        alpha (float): prior probability that target letter appears in presented sequence.

    TODO:
    - the crucial fragile element here is whether the "none" label appears in
      position 0 or position -1 of the classifier output.
    - This is decided when labels were constructed during data prep (and causes the classifier to learn this position).
    - The "none" element of the classifier's output should be spread across the rest of the alphabet.
    - The "(1 - alpha)" element of label prior should be at the same position as the "none" label.

    Output:
        updated vectors of log probabilities - shape (batch, alphabet_len)
    """
    log_likelihoods = compute_log_likelihoods(
        model_log_probs=model_log_probs,
        alphabet_len=log_prior.shape[1],
        presented_seq_idx=presented_seq_idx,
        none_class_idx=none_class_idx,
        alpha=alpha,
    )
    unnormalized_log_posterior = log_likelihoods + log_prior
    return unnormalized_log_posterior - unnormalized_log_posterior.logsumexp(1, keepdim=True)


def compute_log_likelihoods(
    model_log_probs: Tensor,
    alphabet_len: int,
    presented_seq_idx: Tensor,
    none_class_idx: int,
    alpha: float = None,
):
    """
    Given a pretrained classifier, use the provided evidence to update our posterior estimate of the log probabilities.
    Operates on a batch of inputs, each updating a separate alphabet distribution.

    Input:
        alphabet_len: ...
        model_log_probs (Tensor): shape (batch_size, seq_len+1) model log probs for presented letters and "none" class
        presented_seq_idx (Tensor): shape (batch_size, seq_len): indices of letters used, in order of appearance
        none_class_idx (int): 0 or seq_len - the index of the "none" class from classifier's output.
                              This is either the first or last index of the model's `seq_len+1` output values.
        alpha (float): prior probability that target letter appears in presented sequence.

    TODO:
    - the crucial fragile element here is whether the "none" label appears in
      position 0 or position -1 of the classifier output.
    - This is decided when labels were constructed during data prep (and causes the classifier to learn this position).
    - The "none" element of the classifier's output should be spread across the rest of the alphabet.
    - The "(1 - alpha)" element of label prior should be at the same position as the "none" label.

    Output:
        updated vectors of log probabilities - shape (batch, alphabet_len)
    """
    batch_size = model_log_probs.shape[0]
    seq_len = presented_seq_idx.shape[1]
    alpha = alpha if alpha is not None else 0.5
    assert 0 <= alpha <= 1
    assert none_class_idx in [0, seq_len], f"Invalid none_class_idx: {none_class_idx}"

    none_class_probs = model_log_probs[:, none_class_idx]  # shape (batch_size, 1)
    seen_class_probs = torch.cat([model_log_probs[:, :none_class_idx], model_log_probs[:, none_class_idx + 1:]], 1)

    # Compute label likelihood term.
    log_p_label_given_e = (
        # For unseen characters, spread probability mass from the "none" class evenly
        (torch.ones(batch_size, alphabet_len) * none_class_probs[:, None].exp() / (alphabet_len - seq_len)).log()
        # For seen characters, use their output from the model (which is already in log space)
        .scatter(dim=1, index=presented_seq_idx, src=seen_class_probs)
    )

    # Compute label prior term.
    log_p_label = (
        # With prob (1-alpha), target letter is not in sequence. This mass is spread uniformly across all unseen letters
        (torch.ones(batch_size, alphabet_len) * (1 - alpha) / (alphabet_len - seq_len))
        # With prob alpha, target letter is at a uniform random position among seen letters.
        .scatter(dim=1, index=presented_seq_idx, src=torch.tensor(alpha / seq_len).expand_as(presented_seq_idx)).log()
    )

    return log_p_label_given_e - log_p_label


def select_sequence_batch(alphabet_log_probs, seq_len, shuffle=True):
    sequences = []
    for row in alphabet_log_probs:
        sequences.append(select_sequence(row, seq_len, shuffle))
    return torch.stack(sequences)


def select_sequence(alphabet_log_probs, seq_len, shuffle=True):
    sequence = torch.topk(alphabet_log_probs, seq_len).indices
    if shuffle:
        r = torch.randperm(seq_len)
        sequence = sequence[r]
    return sequence
