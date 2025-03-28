from pathlib import Path
from typing import Dict, List, Tuple

from bcipy.config import SESSION_DATA_FILENAME
from bcipy.core.parameters import Parameters
from bcipy.core.session import read_session
from bcipy.core.stimuli import InquirySchedule
from bcipy.language.main import BciPyLanguageModel
from bcipy.signal.model.base_model import SignalModel
from bcipy.simulator.data.sampler.base_sampler import Sampler
from bcipy.simulator.data.sampler.replay_sampler import ReplaySampler
from bcipy.simulator.exceptions import IncompatibleSampler
from bcipy.simulator.task.copy_phrase import SimulatorCopyPhraseTask
from bcipy.simulator.util.state import SimState
from bcipy.task.data import EvidenceType, Inquiry
from bcipy.task.main import TaskData, TaskMode


def clone_inquiry(original_inquiry: Inquiry) -> Inquiry:
    """Construct a new inquiry data record."""
    new_inquiry = Inquiry(
        original_inquiry.stimuli,
        timing=original_inquiry.timing,
        triggers=original_inquiry.triggers,
        target_info=original_inquiry.target_info,
        target_letter=original_inquiry.target_letter,
        current_text=original_inquiry.current_text,
        target_text=original_inquiry.target_text,
        selection=original_inquiry.selection,
        next_display_state=original_inquiry.next_display_state)
    new_inquiry.precision = original_inquiry.precision
    return new_inquiry


class ReplayTask(SimulatorCopyPhraseTask):
    """Task to replay a session and record predictions on that data for a new model."""

    name = "Replay Session"
    mode = TaskMode.COPYPHRASE

    def __init__(self, parameters: Parameters, file_save: str,
                 signal_models: List[SignalModel],
                 language_model: BciPyLanguageModel, samplers: Dict[SignalModel,
                                                                      Sampler]):
        super().__init__(parameters, file_save, signal_models, language_model,
                         samplers)

        if len(signal_models) > 1:
            self.logger.warning("Only the first provided model will be used.")

        self.signal_model = signal_models[0]
        self.sampler = samplers[self.signal_model]

        if not isinstance(self.sampler, ReplaySampler):
            raise IncompatibleSampler(
                "ReplaySampler is the only supported sampler for this task")

    def execute(self) -> TaskData:
        """Executes the task.

        Returns
        -------
        data save location (triggers.txt, session.json)
        """
        # assertion to avoid typing error
        assert isinstance(self.sampler,
                          ReplaySampler), "Sampler must be ReplaySampler"

        self.logger.info("Starting Replay Task!")
        data_folder = self.sampler.current_source
        original_session = read_session(
            str(Path(data_folder, SESSION_DATA_FILENAME)))

        # Iterate through original session.
        for series_n, series in enumerate(original_session.series):
            for inquiry_n, inquiry in enumerate(series):

                # Variables used to generate sim state for sampling.
                self.current_inquiry = InquirySchedule(
                    stimuli=inquiry.stimuli,
                    durations=inquiry.timing,
                    colors=[])
                self.series_n = series_n + 1  # 1-based index
                self.inquiry_n = inquiry_n
                self.inquiry = inquiry

                evidence_types = self.add_evidence(stim_times=[], proceed=True)

                # we only need to evaluate_evidence to involve the copy phrase wrapper
                # for combined likelihood. If that's not used it can be deleted.
                _decision = self.evaluate_evidence()

                data = clone_inquiry(inquiry)
                evidence = self.copy_phrase_task.conjugator.latest_evidence
                data.evidences = {
                    ev_type: ev_data if ev_type in evidence_types else []
                    for ev_type, ev_data in evidence.items()
                }
                data.likelihood = list(
                    self.copy_phrase_task.conjugator.likelihood)

                # Set decision_made to False since we are manually calling add_series.
                self.update_session_data(data, save=True, decision_made=False)
            self.session.add_series()

        self.sampler.next_source()
        self.cleanup()

        return TaskData(save_path=self.file_save,
                        task_dict=self.session.as_dict())

    def compute_device_evidence(
            self,
            stim_times: List[List],
            proceed: bool = True) -> List[Tuple[EvidenceType, List[float]]]:
        """Override to compute the model evidence based on the current state."""
        assert self.signal_model, "Signal model is required"

        current_state = self.get_sim_state()
        self.logger.info("Computing evidence with sim_state:")
        self.logger.info(current_state)

        sampled_data = self.sampler.sample_data(current_state)
        evidence = self.signal_model.compute_likelihood_ratio(
            sampled_data, current_state.display_alphabet, self.alp)
        evidences = [(EvidenceType.ERP, evidence)]

        return evidences

    def get_sim_state(self) -> SimState:
        """Get the current state in the format expected by the simulation tools."""

        return SimState(
            target_symbol=self.inquiry.target_letter or '',
            current_sentence=self.inquiry.current_text or '',
            target_sentence=self.inquiry.target_text or '',
            # without the '+' fixation
            display_alphabet=self.inquiry.stimuli[1:],
            inquiry_n=self.inquiry_n,
            series_n=self.series_n)
