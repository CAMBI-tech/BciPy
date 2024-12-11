"""Defines the Copy Phrase Task which uses a Matrix display"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from bcipy.acquisition.multimodal import ContentType
from bcipy.core.parameters import Parameters
from bcipy.core.stimuli import InquirySchedule
from bcipy.task.control.evidence import EvidenceEvaluator
from bcipy.task.data import EvidenceType
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask


class UpdateData(NamedTuple):
    """Data produced for each inquiry. Sent to the signal model for updating.

    TODO: should data be raw_data, or reshaped?
    """
    preprocessed_data: np.ndarray  # for a single inquiry; transformed and reshaped.
    symbols: List[str]
    times: List[float]
    target_info: List[str]
    window_length: float


class TransferLearningCopyPhraseTask(RSVPCopyPhraseTask):
    """Transfer Learning Copy Phrase Task.

    RSVP Copy Phrase task which does not require initial calibration. Instead,
    it relies on an EEG model that has been previously trained and that can be
    dynamically updated during the course of the task.

    TODO: Questions/Refinements
        - Should we always force the target to be shown?
            There are 2 options for this:
            1. use Oracle Language Model
            2. provide constants to NBestStimuliAgent return_stimuli method
                CopyPhraseWrapper -> DecisionMaker.inq_constants
                called via: copy_phrase_wrapper.decide()
    """
    name = 'Transfer Learning Copy Phrase'

    def __init__(self,
                 parameters: Parameters,
                 file_save: str,
                 fake: bool = False,
                 **kwargs):
        super().__init__(parameters, file_save, fake, **kwargs)

        self.data_cache: Dict[ContentType, List[UpdateData]] = {
            ContentType.EEG: []
        }

    def should_update_model(self) -> bool:
        """Determines whether or not the models get updated. Called at the end
        of each series (after a symbol is spelled).
        """
        return True

    def evaluate(self, evaluator: EvidenceEvaluator,
                 device_data: Dict[ContentType, np.ndarray],
                 symbols: List[str], times: List[float], labels: List[str],
                 window_length: float) -> Tuple[EvidenceType, List[float]]:
        """Evaluate evidence for a single device."""
        content_type = evaluator.consumes
        data = evaluator.preprocess(device_data[content_type], times, labels, window_length)
        data = UpdateData(preprocessed_data=data,
                          symbols=symbols,
                          times=times,
                          target_info=labels,
                          window_length=window_length)
        self.cache(content_type, data)

        return super().evaluate(evaluator, device_data, symbols, times, labels,
                                window_length)

    def next_inquiry(self) -> Optional[InquirySchedule]:
        """Called to initialize a series. Overridden here to also update the models."""
        if self.should_update_model():
            self.update_models()
        return super().next_inquiry()

    def cache(self, content_type: ContentType, data: UpdateData) -> None:
        """Cache the update data to eventually send to the model."""
        if content_type in self.data_cache:
            self.data_cache[content_type].append(data)
        else:
            self.data_cache[content_type] = [data]

    def update_models(self) -> None:
        """Called periodically to update models with the latest cached data."""
        for evaluator in self.evidence_evaluators:
            content_type = evaluator.consumes
            if content_type in self.data_cache and self.data_cache[
                    content_type]:
                update_data = self.data_cache[content_type]
                model = evaluator.signal_model
                # TODO: send data to model for updating
                self.data_cache[content_type].clear()
