"""Simulates the Copy Phrase task"""
from typing import Dict, List, Optional, Tuple

from bcipy.display.main import Display
from bcipy.feedback.visual.visual_feedback import VisualFeedback
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.stimuli import InquirySchedule
from bcipy.language.main import LanguageModel
from bcipy.signal.model.base_model import SignalModel
from bcipy.simulator.helpers.sampler import Sampler
from bcipy.simulator.helpers.state_manager import SimState
from bcipy.simulator.task.null_display import NullDisplay
from bcipy.task.control.evidence import EvidenceEvaluator
from bcipy.task.data import EvidenceType
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask

DEFAULT_EVIDENCE_TYPE = EvidenceType.ERP


def get_evidence_type(model: SignalModel) -> EvidenceType:
    """Get the evidence type provided by the given model"""
    return model.metadata.evidence_type or DEFAULT_EVIDENCE_TYPE


class SimTask():
    """Abstract class that marks a task as a simulation."""
    def __init__(self, parameters: Parameters, file_save: str,
                 signal_models: List[SignalModel],
                 language_model: LanguageModel, samplers: Dict[SignalModel,
                                                               Sampler]):
        """Signature for a simulation task"""

    def execute(self) -> str:
        """Executes the task"""
        raise NotImplementedError


class SimulatorCopyPhraseTask(RSVPCopyPhraseTask, SimTask):
    """CopyPhraseTask that simulates user interactions by sampling data
    from a DataSampler."""

    def __init__(self, parameters: Parameters, file_save: str,
                 signal_models: List[SignalModel],
                 language_model: LanguageModel, samplers: Dict[SignalModel,
                                                               Sampler]):
        super().__init__(win=None,
                         daq=None,
                         parameters=parameters,
                         file_save=file_save,
                         signal_models=signal_models,
                         language_model=language_model,
                         fake=False)
        self.save_session_every_inquiry = False
        self.samplers = samplers

    def init_evidence_evaluators(
            self, signal_models: List[SignalModel]) -> List[EvidenceEvaluator]:
        # Evidence will be sampled so we don't need to evaluate raw data.
        return []

    def init_evidence_types(
            self, signal_models: List[SignalModel],
            evidence_evaluators: List[EvidenceEvaluator]
    ) -> List[EvidenceType]:
        evidence_types = set(
            [get_evidence_type(model) for model in self.signal_models])
        return [EvidenceType.LM, *evidence_types]

    def init_display(self) -> Display:
        return NullDisplay()

    def init_feedback(self) -> Optional[VisualFeedback]:
        return None

    def user_wants_to_continue(self) -> bool:
        return True

    def wait(self, seconds: Optional[float] = None) -> None:
        """Override to do nothing"""

    def present_inquiry(
        self, inquiry_schedule: InquirySchedule
    ) -> Tuple[List[Tuple[str, float]], bool]:
        """Override ; returns empty timing info; always proceed for inquiry
        preview"""
        return [], True

    def show_feedback(self, selection: str, correct: bool = True) -> None:
        """Override to do nothing"""
        # TODO: consider tracking metrics

    def compute_button_press_evidence(
            self, proceed: bool) -> Optional[Tuple[EvidenceType, List[float]]]:
        # TODO: consider adding a sampler for this
        return None

    def compute_device_evidence(
            self,
            stim_times: List[List],
            proceed: bool = True) -> List[Tuple[EvidenceType, List[float]]]:

        current_state = self.get_sim_state()
        self.logger.info("Computing evidence with sim_state:")
        self.logger.info(current_state)

        evidences = []

        for model in self.signal_models:
            sampler = self.samplers[model]
            # This assumes that sampling is independent. Changes to the sampler API are needed if
            # we need to provide the trial context of the last sample.
            sampled_data = sampler.sample_data(current_state)
            evidence = model.predict(sampled_data, self.current_symbols(),
                                     self.alp)
            evidence_type = model.metadata.evidence_type or EvidenceType.ERP
            evidences.append((evidence_type, evidence))
        return evidences

    def exit_display(self) -> None:
        """Close the UI and cleanup."""

    def elapsed_seconds(self) -> float:
        # TODO: estimate this
        return 0.0

    def write_offset_trigger(self) -> None:
        """Do nothing"""

    def write_trigger_data(self, stim_times: List[Tuple[str, float]],
                           target_stimuli: str) -> None:
        """Do nothing"""

    def get_sim_state(self) -> SimState:
        """Get the current state in the format expected by the simulation tools."""

        return SimState(
            target_symbol=self.next_target(),
            current_sentence=self.spelled_text,
            target_sentence=self.copy_phrase,
            display_alphabet=self.current_symbols(),
            # TODO:
            inquiry_n=0,
            series_n=0,
            series_results=[],
            decision_criterion=[])

    def current_symbols(self) -> List[str]:
        """Get the list of symbols from the current inquiry."""
        assert self.current_inquiry, "current inquiry not initialized."
        schedule = self.current_inquiry
        stimuli = schedule.stimuli[0]
        _fixation, *symbols = stimuli
        return symbols
