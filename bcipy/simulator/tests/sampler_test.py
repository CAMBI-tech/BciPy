import random
from pathlib import Path

import numpy as np

from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.symbols import alphabet
from bcipy.signal.model import PcaRdaKdeModel
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.sampler import Sampler, SimpleLetterSampler
from bcipy.simulator.interfaces import SimSessionState

if __name__ == "__main__":
    args = dict()
    args['data_folders'] = ["/Users/srikarananthoju/cambi/tab_test_dynamic/tab_test_dynamic_RSVP_Copy_Phrase_Thu_24_Aug_2023_18hr58min16sec_-0700"]
    args['out_dir'] = Path(__file__).resolve().parent
    model_file = Path(
        "/Users/srikarananthoju/cambi/tab_test_dynamic/tab_test_dynamic_RSVP_Calibration_Thu_24_Aug_2023_18hr41min37sec_-0700/model_0.9524_200_800.pkl")
    sim_parameters = load_json_parameters("bcipy/simulator/sim_parameters.json", value_cast=True)

    data_engine = RawDataEngine(args['data_folders'])
    target_phrase = "HELLO_WORLD"
    target_symbol = "H"

    display_alp = random.sample(alphabet(), 10)
    state: SimSessionState = SimSessionState(target_symbol=target_symbol, inquiry_n=0, series_n=0, target_sentence=target_phrase,
                                             current_sentence="", display_alphabet=display_alp)

    sampler: Sampler = SimpleLetterSampler(data_engine)
    sample: np.ndarray = sampler.sample(state)

    model = PcaRdaKdeModel()
    model = model.load(model_file)

    breakpoint()
    eeg_evidence = model.predict(sample, state.display_alphabet, alphabet())

    print(eeg_evidence.shape)
    print(eeg_evidence)

    breakpoint()
