
class InsufficientDataException(Exception):
    """Insufficient Data Exception.

    Thrown when data requirements to execute task are violated.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


class TaskRegistryException(Exception):
    """Task Registry Exception.

    Thrown when task type is unregistered.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


class MissingEvidenceEvaluator(Exception):
    """Thrown when an evidence evaluator can't be found that matches the
    provided data content type input and evidence_type output"""


class DuplicateModelEvidence(Exception):
    """Thrown from a task when more than one of the provided models produces
    the same type of evidence"""
