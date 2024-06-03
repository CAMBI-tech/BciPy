import os
from bcipy.orchestrator.session_orchestrator import SessionOrchestrator
from bcipy.orchestrator.actions import OfflineAnalysisAction
from bcipy.config import DEFAULT_PARAMETERS_PATH

action = OfflineAnalysisAction('data/analysis_test', DEFAULT_PARAMETERS_PATH)
orchestrator = SessionOrchestrator()

orchestrator.add_task(action)
orchestrator.execute()