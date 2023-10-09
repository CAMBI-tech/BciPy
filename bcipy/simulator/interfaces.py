from abc import ABC

from bcipy.helpers.symbols import alphabet
from bcipy.signal.model import PcaRdaKdeModel
from bcipy.simulator.helpers.sim_state import SimState


class ModelHandler(ABC):

    def generate_evidence(self, state: SimState, features):
        ...

    def get_model(self, key=None):
        ...


class MetricReferee:
    ...


