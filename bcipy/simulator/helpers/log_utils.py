from typing import List

import pandas as pd


def format_alp_likelihoods(likelihoods, alp):
    rounded = [round(lik, 3) for lik in likelihoods]
    formatted = [f"{a} : {l}" for a, l in zip(alp, rounded)]
    return formatted


def format_sample_rows(sample_rows: List[pd.Series]) -> str:
    """Returns a tabular representation of the sample rows."""
    return pd.DataFrame(sample_rows).drop(columns=['eeg']).to_string(
        index=False, header=True)
