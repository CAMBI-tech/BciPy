import random

from bcipy.helpers import load
from bcipy.helpers.symbols import alphabet
from bcipy.simulator.helpers.data_engine import DataEngine
from bcipy.simulator.helpers.sampler import Sampler
from bcipy.simulator.helpers.sim_state import StateManager, SimSessionState
from bcipy.simulator.interfaces import MetricReferee, ModelHandler
from bcipy.simulator.simulator_base import Simulator


class SimulatorCopyPhrase(Simulator):
    """

    Copy Phrase simulator.

    Run loop:
        - The loop runs until StateManager determines simulation is over
        - The generated stimuli are passed into sampler, which returns some eeg response
        - The Model handler will generate predictions and feed back into StateManager
        - StateManager will resolve the state while the MetricReferee observes and records any metrics

    Main components:
        - DataEngine:       loading and storing data
        - Sampler:          logic for sampling from data, composed of DataEngine
        - StateManager:     manages run loop details and fuses model evidence with current state
        - MetricReferee:    tracks and scores the simulation on different metrics for later evaluation
        - ModelHandler:     wrapper for models that deals with loading in models and generating evidence

    Requirements:
    - run closed loop simulation of {TaskType} with {data} with {simulation_params}
    """

    def __init__(self, parameter_path: str, save_dir: str, data_engine: DataEngine, model_handler: ModelHandler, sampler: Sampler,
                 state_manager: StateManager, referee: MetricReferee, verbose=False, visualize=False):
        super(SimulatorCopyPhrase, self).__init__()

        self.parameters = self.load_parameters(parameter_path)
        self.save_dir: str = save_dir
        self.model_handler: ModelHandler = model_handler
        self.sampler: Sampler = sampler
        self.referee: MetricReferee = referee
        self.state_manager: StateManager = state_manager

        self.data_engine: DataEngine = data_engine

        self.symbol_set = alphabet()
        self.write_output = False
        self.data_loader = None

        self.verbose = verbose
        self.visualize = visualize

        # self.signal_models_classes = [PcaRdaKdeModel for m in self.signal_models]  # Hardcoded rn

    def run(self):
        while not self.state_manager.is_done():
            self.state_manager.add_state('display_alphabet', self.__get_inquiry_alp_subset(self.state_manager.get_state()))
            sampled_data = self.sampler.sample(self.state_manager.get_state())
            evidence = self.model_handler.generate_evidence(self.state_manager.get_state(), sampled_data)

            self.state_manager.update(evidence)

        # TODO visualize results

    def __get_inquiry_alp_subset(self, state: SimSessionState):
        # TODO put this in own file or object
        subset_length = 10
        return random.sample(self.symbol_set, subset_length)

    def load_parameters(self, path):
        # TODO validate parameters
        parameters = load.load_json_parameters(path, value_cast=True)
        sim_parameters = load.load_json_parameters(
            "bcipy/simulator/sim_parameters.json", value_cast=True)

        parameters.add_missing_items(sim_parameters)
        return parameters

    def get_param(self, name):
        pass

# TODO add stronger typing hints
