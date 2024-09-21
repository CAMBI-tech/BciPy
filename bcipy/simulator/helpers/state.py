from dataclasses import dataclass
from typing import List


@dataclass
class SimState:
    """ Represents the state of a current session during simulation """
    target_symbol: str
    current_sentence: str
    target_sentence: str
    display_alphabet: List[str]
    inquiry_n: int
    series_n: int
