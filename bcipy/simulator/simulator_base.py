""" base simulator interface """

from abc import ABC, abstractmethod


class Simulator(ABC):
    """Simulator.

    Base class for BciPy Simulations.

    Requirements:
    - run closed loop simulation of {TaskType} with {data} with {simulation_params}
    """

    @abstractmethod
    def run(self):
        """ Run loop for simulation"""

    @abstractmethod
    def get_param(self, name):
        """ retrieving parameter """
