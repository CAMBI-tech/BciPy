
from abc import ABC, abstractmethod


class Simulator(ABC):
    """Simulator.

    Base class for BciPy Simulations.

    Requirements:
    - run closed loop simulation of {TaskType} with {data} with {simulation_params}
    """

    def __init__(self):
        super(Simulator, self).__init__()

    @abstractmethod
    def run(self):
        ...

    @abstractmethod
    def get_param(self, name):
        ...
