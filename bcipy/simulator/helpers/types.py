from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np

from bcipy.task.data import EvidenceType


@dataclass
class SimEvidence:
    evidence_type: EvidenceType
    evidence: np.ndarray
    symbol_set: List[str]


@dataclass
class InquiryResult:
    target: Optional[str]
    time_spent: int
    stimuli: List
    evidences: Dict[str, SimEvidence]
    fused_likelihood: np.ndarray
    decision: Optional[str]
