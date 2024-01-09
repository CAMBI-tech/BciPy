from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from bcipy.task.data import EvidenceType


@dataclass
class InquiryResult:
    target: Optional[str]
    time_spent: int
    stimuli: List
    # TODO make this into a dictionary to support multimodal. e.g
    # {SignalModel: evidence_list, LanguageModel:evidence_list}
    evidence_likelihoods: List
    fused_likelihood: np.ndarray
    decision: Optional[str]


@dataclass
class SimEvidence:
    evidence_type: EvidenceType
    evidence: np.ndarray
    symbol_set: List[str]
