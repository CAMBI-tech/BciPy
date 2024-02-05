from typing import List, Dict

import pandas as pd

from bcipy.simulator.helpers.types import SimEvidence


def fmt_stim_likelihoods(likelihoods, alp):
    rounded = [round(lik, 3) for lik in likelihoods]
    formatted = [f"{a} : {l}" for a, l in zip(alp, rounded)]
    return formatted


def fmt_likelihoods_for_hist(likelihoods, alp):
    rounded = [round(lik, 3) for lik in likelihoods]
    formatted = [(a, l) for a, l in zip(alp, rounded)]
    return formatted


def format_sample_rows(sample_rows: List[pd.DataFrame]):
    formatted_rows = []
    for row in sample_rows:
        new_row = row.drop(columns=['eeg'], axis=1, inplace=False)
        formatted_rows.append(new_row.to_string(index=False, header=True))

    return ", ".join(formatted_rows)


def fmt_reshaped_evidence(evidences: Dict[str, SimEvidence]):
    """ Formats evidences to log shapes pf ndarrays """

    evidence_shape_strs = []

    for evidence_name, evidence in evidences.items():
        evidence_shape = evidence.evidence.shape if evidence is not None else None
        evidence_shape_strs.append(f"{evidence.evidence_type} => {evidence_shape}")

    return ", ".join(evidence_shape_strs)
