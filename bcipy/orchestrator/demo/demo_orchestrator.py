from bcipy.config import DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETERS_PATH
from bcipy.orchestrator import SessionOrchestrator
from bcipy.task.actions import (ExperimentFieldCollectionAction,
                                OfflineAnalysisAction)
from bcipy.task.registry import TaskRegistry
from bcipy.task.paradigm.rsvp import RSVPCalibrationTask, RSVPCopyPhraseTask, RSVPTimingVerificationCalibration
from bcipy.task.paradigm.matrix import MatrixCalibrationTask

def demo_orchestrator(parameters_path: str) -> None:
    """Demo the SessionOrchestrator.

    This function demonstrates how to use the SessionOrchestrator to execute actions.

    The action in this case is an OfflineAnalysisAction, which will analyze the data in a given directory.
    """
    # field_collection = ExperimentFieldCollectionAction(DEFAULT_EXPERIMENT_ID, data_path)
    # offline_analysis = OfflineAnalysisAction(data_path, parameters_path)
    tasks = [RSVPCalibrationTask, OfflineAnalysisAction]
    orchestrator = SessionOrchestrator(parameters_path=parameters_path, fake=True)
    orchestrator.add_task(RSVPTimingVerificationCalibration)
    orchestrator.add_task(RSVPCalibrationTask)
    orchestrator.add_task(MatrixCalibrationTask)
    orchestrator.add_task(RSVPCopyPhraseTask)
    orchestrator.add_task(OfflineAnalysisAction)
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
