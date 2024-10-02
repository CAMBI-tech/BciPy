from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.task.orchestrator import SessionOrchestrator
from bcipy.task.actions import (OfflineAnalysisAction, IntertaskAction)
from bcipy.task.paradigm.rsvp import RSVPCalibrationTask, RSVPCopyPhraseTask, RSVPTimingVerificationCalibration
from bcipy.task.paradigm.matrix.timing_verification import MatrixTimingVerificationCalibration


def demo_orchestrator(parameters_path: str) -> None:
    """Demo the SessionOrchestrator.

    This function demonstrates how to use the SessionOrchestrator to execute actions.

    The action in this case is an OfflineAnalysisAction, which will analyze the data in a given directory.
    """
    tasks = [
        MatrixTimingVerificationCalibration,
        IntertaskAction,
        RSVPTimingVerificationCalibration]
    orchestrator = SessionOrchestrator(
        user='time_test_orchestrator',
        parameters_path=parameters_path,
        alert=True,
        fake=True)
    orchestrator.add_tasks(tasks)
    orchestrator.execute()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Demo the SessionOrchestrator")
    parser.add_argument(
        '-p',
        '--parameters_path',
        help='Path to the parameters file to use for training.  If none provided, data path will be used.',
        default=DEFAULT_PARAMETERS_PATH)
    args = parser.parse_args()

    parameters_path = f'{args.parameters_path}'

    demo_orchestrator(parameters_path)
