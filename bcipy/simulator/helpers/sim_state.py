from abc import ABC
from dataclasses import dataclass


@dataclass
class SimSessionState:
    """ Represents the state of a current session during simulation """
    target_symbol: str
    current_sentence: str
    target_sentence: str
    display_alphabet: list[str]
    inquiry_n: int
    series_n: int


class StateManager(ABC):

    def update(self, evidence):
        ...

    def is_done(self) -> bool:
        ...

    def get_state(self) -> SimSessionState:
        ...

    def add_state(self, state_field, state_value):
        pass
