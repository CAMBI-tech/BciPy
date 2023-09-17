from typing import Optional

import numpy as np

from bcipy.config import SESSION_DATA_FILENAME
from bcipy.helpers import load
from bcipy.helpers.list import grouper
from bcipy.helpers.session import evidence_records, read_session
from bcipy.helpers.symbols import alphabet
from bcipy.signal.model import PcaRdaKdeModel
from bcipy.simulator.helpers.signal_helpers import process_raw_data_for_model, ExtractedExperimentData
from bcipy.simulator.helpers.sim_viz import plot_replay_comparison
from bcipy.simulator.simulator_base import Simulator, SimulatorData


class SimulatorCopyPhraseReplay(Simulator):
    """Simulator.

    Base class for BciPy Simulations.

    Requirements:
    - run closed loop simulation of {TaskType} with {data} with {simulation_params}
    """

    def __init__(self, parameter_path, save_dir, signal_models: [tuple], language_models, replay_data_path=None,
                 verbose=False):
        super(SimulatorCopyPhraseReplay, self).__init__()

        self.parameters = self.load_parameters(parameter_path)
        self.save_dir = save_dir
        self.signal_models = signal_models
        self.language_models = language_models
        self.replay_data_path = replay_data_path

        self.symbol_set = alphabet()
        self.write_output = False
        self.data_loader = None

        self.signal_models_classes = [PcaRdaKdeModel for m in self.signal_models] # Hardcoded rn

    def run(self):
        self.init_data_loader()

        self.logger.info(f"Initialized data loader")

        # TODO need to support multimodal and language models
        model_file, model_class = self.signal_models[0], self.signal_models_classes[0] # hardcoded rn
        model = model_class(self.parameters.get("k_folds"))
        model = model.load(model_file)

        self.logger.info(f"Loaded model from {model_file}, with model_class {model_class}")

        target_eeg_evidences, non_target_eeg_evidences = [], []
        while not self.data_loader.is_end():
            inquiry_trials, this_inquiry_letters, this_inquiry_labels = self.data_loader.next()
            target_eeg_evidence, non_target_eeg_evidence = [], []
            response = model.predict(inquiry_trials, this_inquiry_letters, symbol_set=self.symbol_set)

            symbol_letter_idx_map = [self.symbol_set.index(letter) for letter in this_inquiry_letters]
            for i, label in enumerate(this_inquiry_labels):
                if label:
                    target_eeg_evidence.append(response[symbol_letter_idx_map[i]])
                else:
                    non_target_eeg_evidence.append(response[symbol_letter_idx_map[i]])

            target_eeg_evidences.append(target_eeg_evidence)
            non_target_eeg_evidences.append(non_target_eeg_evidence)

        target_eeg_evidences, non_target_eeg_evidences = np.concatenate(target_eeg_evidences), np.concatenate(
            non_target_eeg_evidences)

        # Get target and non_target eeg values from actual experiment in session.json for comparison
        session_records = evidence_records(read_session(self.replay_data_path / SESSION_DATA_FILENAME))
        session_target_eeg = np.array([record.eeg for record in session_records if record.is_target])
        session_non_target_eeg = np.array([record.eeg for record in session_records if not record.is_target])

        # note this flattens any grouping of inquiries which might be problem for future visualizations

        # visualizing comparison of results
        plot_replay_comparison(target_eeg_evidences, non_target_eeg_evidences, session_target_eeg,
                               session_non_target_eeg, self.save_dir, self.parameters)

        self.logger.info(f"Plotted comparison charts "
                         f"\n New: {len(target_eeg_evidences)} target_eeg_evidences | {len(non_target_eeg_evidences)} non_target_eeg_evidences "
                         f"\n Old: {len(session_target_eeg)} target_eeg_evidences | {len(session_non_target_eeg)} non_target_eeg_evidences ")

    def get_param(self, name):
        return self.parameters.get(name, None)

    def init_data_loader(self):

        if self.replay_data_path:
            self.data_loader = ReplayDataLoader(self.replay_data_path, self.parameters)
        else:
            pass  # TODO sampling data logic
        pass

    def load_parameters(self, path):
        # TODO validate parameters
        parameters = load.load_json_parameters(path, value_cast=True)
        sim_parameters = load.load_json_parameters(
            "bcipy/simulator/sim_parameters.json", value_cast=True)

        parameters.add_missing_items(sim_parameters)
        return parameters


class ReplayDataLoader(SimulatorData):

    def __init__(self, source_path, parameters):
        self.source_path = source_path
        self.parameters = parameters
        self.verbose = False

        self.inquiry_worth_of_letters = None
        self.inquiry_worth_of_trials = None
        self.data: Optional[ExtractedExperimentData] = None

        self.load()

        self.i = 0

    def load(self):
        experiment_data = process_raw_data_for_model(self.source_path, self.parameters)

        # formatting data into inquiries
        trigger_targetness, trigger_timing, trigger_symbols = experiment_data.decoded_triggers
        self.inquiry_worth_of_trials = np.split(experiment_data.trials, experiment_data.inquiries.shape[1], 1)
        self.inquiry_worth_of_letters = [list(group) for group in
                                         grouper(trigger_symbols, self.parameters.get('stim_length'),
                                                 incomplete="ignore")]

        self.data = experiment_data

    def next(self):
        if self.i < len(self.inquiry_worth_of_letters):
            cur_trials = self.inquiry_worth_of_trials[self.i]
            cur_letters = self.inquiry_worth_of_letters[self.i]
            cur_labels = self.data.labels[self.i]

            self.i += 1

            return cur_trials, cur_letters, cur_labels  # TODO turn into dataclass

        return None, None, None

    def is_end(self):
        return self.i >= len(self.inquiry_worth_of_letters)
