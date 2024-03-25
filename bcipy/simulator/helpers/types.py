import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np

from bcipy.task.data import EvidenceType


@dataclass
class SimEvidence:
    evidence_type: EvidenceType
    evidence: np.ndarray
    symbol_set: List[str]

    def to_json(self) -> Dict:
        d = dataclasses.asdict(self)
        evidence_list = self.evidence.tolist()
        d['evidence'] = evidence_list
        d['evidence_type'] = self.evidence_type.name

        return d


@dataclass
class InquiryResult:
    target: Optional[str]
    time_spent: int
    stimuli: List
    evidences: Dict[str, SimEvidence]
    fused_likelihood: np.ndarray
    decision: Optional[str]

    def to_json(self):
        d = dataclasses.asdict(self)

        d['fused_likelihood'] = self.fused_likelihood.tolist()
        d['evidences'] = {k: v.to_json() for k, v in self.evidences.items()}

        return d
