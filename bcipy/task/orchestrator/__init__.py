"""Task orchestration module for managing BCI experiment sessions.

This module provides functionality for managing and executing sequences of BCI tasks,
including task initialization, execution order, data saving, and logging. The main
component is the SessionOrchestrator, which handles the lifecycle of task execution.
"""

from bcipy.task.orchestrator.orchestrator import SessionOrchestrator

__all__ = ['SessionOrchestrator']
