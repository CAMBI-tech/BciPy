from typing import Dict, List

import pandas as pd

from bcipy.simulator.helpers.data_engine import Trial
from bcipy.simulator.helpers.types import SimEvidence


def fmt_stim_likelihoods(likelihoods, alp):
    rounded = [round(lik, 3) for lik in likelihoods]
    formatted = [f"{a} : {l}" for a, l in zip(alp, rounded)]
    return formatted


def fmt_likelihoods_for_hist(likelihoods, alp):
    rounded = [round(lik, 3) for lik in likelihoods]
    formatted = [(a, l) for a, l in zip(alp, rounded)]
    return formatted


def format_samples(sample_rows: List[Trial]) -> str:
    """Returns a tabular representation of the sample rows."""
    return '\n'.join([str(row) for row in sample_rows])

def format_sample_rows(sample_rows: List[pd.Series]) -> str:
    """Returns a tabular representation of the sample rows."""
    return pd.DataFrame(sample_rows).drop(columns=['eeg']).to_string(
        index=False, header=True)


def format_sample_df(sample_rows: pd.DataFrame) -> str:
    """Returns a tabular representation of the sample rows."""
    return sample_rows.drop(columns=['eeg']).to_string(index=False,
                                                       header=True)


def fmt_reshaped_evidence(evidences: Dict[str, SimEvidence]):
    """ Formats evidences to log shapes pf ndarrays """

    evidence_shape_strs = []

    for evidence_name, evidence in evidences.items():
        evidence_shape = evidence.evidence.shape if evidence is not None else None
        evidence_shape_strs.append(f"{evidence.evidence_type} => {evidence_shape}")

    return ", ".join(evidence_shape_strs)
