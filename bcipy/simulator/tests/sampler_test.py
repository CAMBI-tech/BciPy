import argparse
import random
from pathlib import Path
import numpy as np

from bcipy.config import DEFAULT_PARAMETER_FILENAME
from bcipy.helpers import load
from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.symbols import alphabet
from bcipy.signal.model import PcaRdaKdeModel
from bcipy.simulator.helpers.sampler import Sampler, SimpleLetterSampler
from bcipy.simulator.interfaces import SimSessionState
from bcipy.simulator.sim_sampler import RawDataEngine

if __name__ == "__main__":
    args = dict()
    args['data_folders'] = ["/Users/srikarananthoju/cambi/tab_test_dynamic/tab_test_dynamic_RSVP_Copy_Phrase_Thu_24_Aug_2023_18hr58min16sec_-0700"]
    args['parameter_path'] = [load.load_json_parameters(str(Path(data_folder, DEFAULT_PARAMETER_FILENAME)), value_cast=True) for data_folder in
                              args['data_folders']]
    args['out_dir'] = Path(__file__).resolve().parent
    model_file = Path(
        "/Users/srikarananthoju/cambi/tab_test_dynamic/tab_test_dynamic_RSVP_Calibration_Thu_24_Aug_2023_18hr41min37sec_-0700/model_0.9524_200_800.pkl")
    sim_parameters = load_json_parameters("bcipy/simulator/sim_parameters.json", value_cast=True)

    data_engine = RawDataEngine(args['data_folders'], args['parameter_path'])
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
