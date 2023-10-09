from dataclasses import dataclass
from typing import Optional, List


@dataclass
class InquiryResult:
    target: Optional[str]
    time_spent: int  # TODO what does time_spent mean?
    stimuli: List
    evidence_likelihoods: List
    decision: Optional[str]
