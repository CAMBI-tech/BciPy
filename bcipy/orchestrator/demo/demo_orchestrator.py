from bcipy.orchestrator.orchestrator import SessionOrchestrator
from bcipy.orchestrator.actions import OfflineAnalysisAction, ExperimentFieldCollectionAction
from bcipy.config import DEFAULT_EXPERIMENT_ID
from bcipy.config import DEFAULT_PARAMETER_FILENAME

from bcipy.helpers.load import load_experimental_data


def demo_orchestrator(data_path: str, parameters_path: str) -> None:
    """Demo the SessionOrchestrator.

    This function demonstrates how to use the SessionOrchestrator to execute actions.

    The action in this case is an OfflineAnalysisAction, which will analyze the data in a given directory.
    """
    field_collection = ExperimentFieldCollectionAction(DEFAULT_EXPERIMENT_ID, data_path)
    offline_analysis = OfflineAnalysisAction(data_path, parameters_path)
    orchestrator = SessionOrchestrator()
    orchestrator.add_task(field_collection)
    orchestrator.add_task(offline_analysis)
    orchestrator.execute()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Demo the SessionOrchestrator")
    parser.add_argument(
        '-d',
        '--data_path',
        help='Path to the data directory. If none provided, a GUI will open and prompt a choice.',
        default=None)
    parser.add_argument(
        '-p',
        '--parameters_path',
        help='Path to the parameters file to use for training.  If none provided, data path will be used.',
        default=None)
    args = parser.parse_args()
    data_path = args.data_path
    parameters_path = args.parameters_path

    if not data_path:
        data_path = load_experimental_data()

    if not parameters_path:
        parameters_path = f'{data_path}/{DEFAULT_PARAMETER_FILENAME}'

    demo_orchestrator(data_path, parameters_path)
