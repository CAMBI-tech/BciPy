# mypy: disable-error-code="union-attr"
"""Simulates the Copy Phrase task"""
from typing import Any, Dict, List, Optional, Tuple
import logging

from bcipy.display.main import Display
from bcipy.feedback.visual.visual_feedback import VisualFeedback
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.stimuli import InquirySchedule
from bcipy.language.main import LanguageModel
from bcipy.signal.model.base_model import SignalModel
from bcipy.simulator.data.sampler import Sampler
from bcipy.simulator.task.null_display import NullDisplay
from bcipy.simulator.util.state import SimState
from bcipy.task.control.evidence import EvidenceEvaluator
from bcipy.task.data import EvidenceType
from bcipy.task import TaskMode
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask

DEFAULT_EVIDENCE_TYPE = EvidenceType.ERP


def get_evidence_type(model: SignalModel) -> EvidenceType:
    """Get the evidence type provided by the given model and return it as an enum (EvidenceType).

    If the model does not provide an evidence type, the default evidence type is returned (ERP).
    """
    evidence_type = model.metadata.evidence_type

    # Note: once more than one evidence type is supported, this should be updated to throw an error
    if not evidence_type:
        return DEFAULT_EVIDENCE_TYPE

    try:
        return EvidenceType(evidence_type)
    except ValueError:
        raise ValueError(f"Unsupported evidence type: {evidence_type}. Supported types: {EvidenceType.list()}")


class SimulatorCopyPhraseTask(RSVPCopyPhraseTask):
    """CopyPhraseTask that simulates user interactions by sampling data
    from a DataSampler."""

    name = "Simulator Copy Phrase"
    paradigm = "RSVP"
    mode = TaskMode.COPYPHRASE

    def __init__(self, parameters: Parameters, file_save: str,
                 signal_models: List[SignalModel],
                 language_model: LanguageModel, samplers: Dict[SignalModel,
                                                               Sampler]):
        self.args_sm = signal_models
        self.args_lm = language_model
        self.save_session_every_inquiry = False
        self.samplers = samplers
        self.logger = logging.getLogger(__name__)
        super().__init__(parameters=parameters,
                         file_save=file_save,
                         fake=False)

    def setup(
            self,
            parameters: Parameters,
            data_save_location: str,
            fake: bool = False) -> Tuple[Any, Any, Display]:
        """Override the setup method to avoid initializing the data acquisition."""
        daq = None
        server = None
        display = self.init_display()
        self.initalized = True
        return daq, server, display

    def get_signal_models(self):
        """Override the get the signal models"""
        return self.args_sm

    def get_language_model(self):
        """Override the get the language model"""
        return self.args_lm

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

    def compute_button_press_evidence(
            self, proceed: bool) -> Optional[Tuple[EvidenceType, List[float]]]:
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
            evidence = model.compute_class_probabilities(
                sampled_data,
                self.current_symbols(),
                self.alp)
            evidence_type = get_evidence_type(model)
            evidences.append((evidence_type, evidence))
        return evidences

    def exit_display(self) -> None:
        """Close the UI and cleanup."""

    def elapsed_seconds(self) -> float:
        return 0.0

    def write_offset_trigger(self) -> None:
        """Do nothing"""

    def write_trigger_data(self, stim_times: List[Tuple[str, float]],
                           target_stimuli: str) -> None:
        """Do nothing"""

    def get_sim_state(self) -> SimState:
        """Get the current state in the format expected by the simulation tools."""

        return SimState(target_symbol=self.next_target(),
                        current_sentence=self.spelled_text,
                        target_sentence=self.copy_phrase,
                        display_alphabet=self.current_symbols(),
                        inquiry_n=len(self.session.series[-1]),
                        series_n=len(self.session.series))

    def current_symbols(self) -> List[str]:
        """Get the list of symbols from the current inquiry."""
        assert self.current_inquiry, "current inquiry not initialized."
        schedule = self.current_inquiry
        stimuli = schedule.stimuli[0]
        _fixation, *symbols = stimuli
        return symbols
