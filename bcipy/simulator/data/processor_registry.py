"""Registry for data processors."""
import logging as logger
from typing import List, Optional, Type

from bcipy.acquisition.multimodal import ContentType
from bcipy.signal.model.base_model import SignalModel
# flake8: noqa
from bcipy.simulator.data.data_process import (EEGRawDataProcessor,
                                               RawDataProcessor)
from bcipy.simulator.data.switch_data_processor import SwitchDataProcessor
from bcipy.task.data import EvidenceType

log = logger.getLogger(__name__)

PROCESSOR_TYPES = [EEGRawDataProcessor, SwitchDataProcessor]


def get_processors() -> List[Type[RawDataProcessor]]:
    """Returns a list of all available raw data processors."""
    return PROCESSOR_TYPES


def register_processor(processor_type: Type[RawDataProcessor]) -> None:
    """Register a new data processor to be used in a simulation."""
    if not processor_type in PROCESSOR_TYPES:
        PROCESSOR_TYPES.append(processor_type)


# Module functions for matching a model to the correct processor.
def get_processor(
        data_source: ContentType,
        evidence_type: Optional[EvidenceType] = None
) -> Type[RawDataProcessor]:
    """Returns the matching processor class.

    Parameters
    ----------
        data_source - type of data that the processor should consume
        evidence_type - type of evidence that the processor should produce.
    """
    matches = [
        cls for cls in PROCESSOR_TYPES if cls.consumes == data_source and (
            evidence_type is None or cls.produces == evidence_type)
    ]
    if matches:
        return matches[0]
    else:
        msg = f"Data processor not found for {data_source.name}"
        if evidence_type:
            msg += f" -> {evidence_type.name}"
        raise Exception(msg)


def find_data_processor(model: SignalModel) -> Type[RawDataProcessor]:
    """Get the DataProcessor appropriate for the given model."""
    content_type = ContentType(model.metadata.device_spec.content_type)
    # Metadata may provide an EvidenceType with a model so the same data source can
    # be used to produce multiple types of evidence (ex. alpha)
    evidence_type = None
    model_output = model.metadata.evidence_type
    if model_output:
        try:
            evidence_type = EvidenceType(model_output.upper())
        except ValueError:
            logger.error(f"Unsupported evidence type: {model_output}")

    return get_processor(content_type, evidence_type)


def init_data_processor(signal_model: SignalModel) -> RawDataProcessor:
    """Find an DataProcessor that matches the given signal_model and
    initialize it."""
    processor_class = find_data_processor(signal_model)
    return processor_class(signal_model)
