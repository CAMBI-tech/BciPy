import random
from pathlib import Path

import numpy as np

from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.symbols import alphabet
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.metrics import DummyRef
from bcipy.simulator.helpers.sampler import Sampler, EEGByLetterSampler
from bcipy.simulator.helpers.state_manager import StateManagerImpl, StateManager, SimState
from bcipy.simulator.helpers.model_handler import SignalModelHandler1
from bcipy.simulator.sim import SimulatorCopyPhrase



if __name__ == "__main__":
    args = dict()
    args['data_folders'] = ["/Users/srikarananthoju/cambi/tab_test_dynamic/16sec_-0700",
                            "/Users/srikarananthoju/cambi/tab_test_dynamic/50sec_-0700",
                            # "/Users/srikarananthoju/cambi/tab_test_dynamic/29sec_-0700"
                            ]
    args['out_dir'] = Path(__file__).resolve().parent
    model_file = Path(
        "/Users/srikarananthoju/cambi/tab_test_dynamic/calibr_37sec_-0700/model_0.9524_200_800.pkl"
        # "/Users/srikarananthoju/cambi/tab_test_dynamic/calibr_37sec_-0700/model_0.9595.pkl"
    )
    sim_parameters = load_json_parameters("bcipy/simulator/sim_parameters.json", value_cast=True)

    data_engine = RawDataEngine(args['data_folders'])

    display_alp = random.sample(alphabet(), 10)

    stateManager: StateManager = StateManagerImpl(sim_parameters)
    stateManager.mutate_state("display_alphabet", display_alp)
    print(stateManager.get_state())

    sampler: Sampler = EEGByLetterSampler(data_engine)
    sample: np.ndarray = sampler.sample(stateManager.get_state())

    model_handler = SignalModelHandler1(model_file)
    sim = SimulatorCopyPhrase(data_engine, model_handler, sampler, stateManager, DummyRef())
    sim.run()
