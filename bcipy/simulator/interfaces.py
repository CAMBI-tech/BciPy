from abc import ABC
from dataclasses import dataclass


@dataclass
class SessionState:
    target_symbol: str
    inquiry_n: int
    series_n: int
    target_sentence: str
    current_sentence: str
    display_alphabet: list[str]


class StateManager(ABC):

    def update(self, evidence):
        ...

    def is_done(self) -> bool:
        ...

    def get_state(self) -> SessionState:
        ...

    def add_state(self, state_field, state_value):
        pass


class ModelHandler(ABC):

    def generate_evidence(self, state: SessionState, features):
        ...

    def get_model(self, key=None):
        ...


class Sampler(ABC):

    def sample(self, state: SessionState):
        ...


class MetricReferee:
    ...


class DataEngine(ABC):

    def load(self):
        ...

    def transform(self):
        ...

    def get_data(self):
        ...
