""" base simulator interface """

from abc import ABC, abstractmethod
from typing import Optional

from bcipy.helpers.parameters import Parameters
from bcipy.simulator.helpers import metrics
from bcipy.simulator.helpers.data_engine import DataEngine
from bcipy.simulator.helpers.model_handler import ModelHandler
from bcipy.simulator.helpers.sampler import Sampler
from bcipy.simulator.helpers.state_manager import StateManager


class Simulator(ABC):
    """Simulator.

    Base class for BciPy Simulations.

    Requirements:
    - run closed loop simulation of {TaskType} with {data} with {simulation_params}
    """

    model_handler: ModelHandler
    sampler: Sampler
    state_manager: StateManager
    data_engine: DataEngine
    referee: metrics.MetricReferee
    parameters: Parameters
    save_dir: Optional[str]

    @abstractmethod
    def run(self):
        """ Run loop for simulation"""

    @abstractmethod
    def get_parameters(self) -> Parameters:
        """ retrieving parameters copy"""

    def reset(self):
        ...
