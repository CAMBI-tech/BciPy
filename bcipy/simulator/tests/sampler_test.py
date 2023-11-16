import random
from pathlib import Path

import numpy as np

from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.symbols import alphabet
from bcipy.signal.model import PcaRdaKdeModel
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.sampler import Sampler, SimpleLetterSampler
from bcipy.simulator.helpers.state_manager import StateManagerImpl, StateManager, SimState
from bcipy.simulator.interfaces import ModelHandler, MetricReferee
from bcipy.simulator.sim import SimulatorCopyPhrase


class DummyModelHandler(ModelHandler):

    def __init__(self, model_file):
        self.model_file = model_file

    def generate_evidence(self, state: SimState, features):
        model = PcaRdaKdeModel()
        model = model.load(self.model_file)

        stimuli = state.display_alphabet
        alp = alphabet()
        eeg_evidence = model.predict(features, stimuli, alp)

        return eeg_evidence

    def get_model(self, key=None):
        pass


class DummyRef(MetricReferee):
    pass


if __name__ == "__main__":
    args = dict()
    args['data_folders'] = ["/Users/srikarananthoju/cambi/tab_test_dynamic/tab_test_dynamic_RSVP_Copy_Phrase_Thu_24_Aug_2023_18hr58min16sec_-0700",
                            # "/Users/srikarananthoju/cambi/tab_test_dynamic/tab_test_dynamic_RSVP_Copy_Phrase_Thu_24_Aug_2023_19hr07min50sec_-0700",
                            # "/Users/srikarananthoju/cambi/tab_test_dynamic/tab_test_dynamic_RSVP_Copy_Phrase_Thu_24_Aug_2023_19hr15min29sec_-0700"
                            ]
    args['out_dir'] = Path(__file__).resolve().parent
    model_file = Path(
        "/Users/srikarananthoju/cambi/tab_test_dynamic/tab_test_dynamic_RSVP_Calibration_Thu_24_Aug_2023_18hr41min37sec_-0700/model_0.9524_200_800.pkl")
    sim_parameters = load_json_parameters("bcipy/simulator/sim_parameters.json", value_cast=True)

    data_engine = RawDataEngine(args['data_folders'])

    display_alp = random.sample(alphabet(), 10)

    stateManager: StateManager = StateManagerImpl(sim_parameters)
    stateManager.mutate_state("display_alphabet", display_alp)
    print(stateManager.get_state())

    sampler: Sampler = SimpleLetterSampler(data_engine)
    sample: np.ndarray = sampler.sample(stateManager.get_state())

    # model = PcaRdaKdeModel()
    # model = model.load(model_file)
    #
    # eeg_evidence = model.predict(sample, stateManager.get_state().display_alphabet, alphabet())
    #
    # print(eeg_evidence.shape)
    # print(eeg_evidence)

    model_handler = DummyModelHandler(model_file)
    sim = SimulatorCopyPhrase(data_engine, model_handler, sampler, stateManager, DummyRef())
    sim.run()
