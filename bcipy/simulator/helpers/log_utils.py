from typing import List

import pandas as pd


def fmt_stim_likelihoods(likelihoods, alp):
    rounded = [round(lik, 3) for lik in likelihoods]
    formatted = [f"{a} : {l}" for a, l in zip(alp, rounded)]
    return formatted


def fmt_fused_likelihoods_for_hist(likelihoods, alp):
    rounded = [round(lik, 3) for lik in likelihoods]
    formatted = [(a, l) for a, l in zip(alp, rounded)]
    return formatted


def format_sample_rows(sample_rows: List[pd.DataFrame]):
    formatted_rows = []
    for row in sample_rows:
        new_row = row.drop(columns=['eeg'], axis=1, inplace=False)
        formatted_rows.append(new_row.to_string(index=False, header=True))

    return ", ".join(formatted_rows)
