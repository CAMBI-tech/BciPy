from abc import ABC
from bcipy.simulator.helpers.sim_state import SimSessionState


class ModelHandler(ABC):

    def generate_evidence(self, state: SimSessionState, features):
        ...

    def get_model(self, key=None):
        ...


class MetricReferee:
    ...
