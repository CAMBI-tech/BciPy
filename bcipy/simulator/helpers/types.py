from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np

from bcipy.task.data import EvidenceType


@dataclass
class InquiryResult:
    target: Optional[str]
    time_spent: int
    stimuli: List
    evidence_likelihoods: List  # TODO make this into a dictionary to support multimodal. e.g {SignalModel: evidence_list, LanguageModel:evidence_list}
    fused_likelihood: np.ndarray
    decision: Optional[str]


@dataclass
class SimEvidence:
    evidence_type: EvidenceType
    evidence: np.ndarray
    symbol_set: List[str]
