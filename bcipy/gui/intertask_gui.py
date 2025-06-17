"""Intertask GUI module.

This module provides a graphical user interface for managing task transitions
in BciPy experiments, showing progress and allowing users to control task flow.
"""

import logging
from typing import Callable, List, Optional

from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QProgressBar,
                             QPushButton)

from bcipy.config import SESSION_LOG_FILENAME
from bcipy.gui.bciui import BCIUI, run_bciui

logger = logging.getLogger(SESSION_LOG_FILENAME)


class IntertaskGUI(BCIUI):
    """GUI for managing transitions between tasks in an experiment.

    This class provides a progress interface that shows the current task progress
    and allows users to proceed to the next task or stop the experiment.

    Attributes:
        action_name (str): Name of the action type.
        tasks (List[str]): List of task names in the experiment.
        current_task_index (int): Index of the current task.
        next_task_name (str): Name of the next task to be executed.
        total_tasks (int): Total number of tasks in the experiment.
        task_progress (int): Current progress through the tasks.
        callback (Callable): Function to call when stopping tasks.
    """

    action_name = "IntertaskAction"

    def __init__(
        self,
        next_task_index: int,
        tasks: List[str],
        exit_callback: Callable[[], None],
    ) -> None:
        """Initialize the intertask GUI.

        Args:
            next_task_index (int): Index of the next task to be executed.
            tasks (List[str]): List of task names in the experiment.
            exit_callback (Callable[[], None]): Function to call when stopping tasks.
        """
        self.tasks = tasks
        self.current_task_index = next_task_index
        self.next_task_name = tasks[self.current_task_index]
        self.total_tasks = len(tasks)
        self.task_progress = next_task_index
        self.callback = exit_callback
        super().__init__("Progress", 800, 150)
        self.setProperty("class", "inter-task")

    def app(self) -> None:
        """Initialize and configure the GUI application.

        Sets up the progress display, next task information, and control buttons.
        """
        self.contents.addLayout(BCIUI.centered(QLabel("Experiment Progress")))

        progress_container = QHBoxLayout()
        progress_container.addWidget(
            QLabel(f"({self.task_progress}/{self.total_tasks})")
        )
        self.progress = QProgressBar()
        self.progress.setValue(int(self.task_progress / self.total_tasks * 100))
        self.progress.setTextVisible(False)
        progress_container.addWidget(self.progress)
        self.contents.addLayout(progress_container)

        next_info = QHBoxLayout()
        next_info.addWidget(QLabel("Next Task: "))
        next_task = QLabel(self.next_task_name)
        next_task.setStyleSheet("font-weight: bold; color: green;")
        next_info.addWidget(next_task)
        self.contents.addLayout(next_info)

        self.contents.addStretch(1)
        self.next_button = QPushButton("Next")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("background-color: red")
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.next_button)
        self.contents.addLayout(buttons_layout)

        self.next_button.clicked.connect(self.next)
        self.stop_button.clicked.connect(self.stop_tasks)

    def stop_tasks(self) -> None:
        """Stop the current task execution.

        Calls the exit callback and quits the application.
        """
        # This should exit Task executions
        logger.info(f"Stopping Tasks... user requested. Using callback: {self.callback}")
        self.callback()
        self.quit()
        logger.info("Tasks Stopped")

    def next(self) -> None:
        """Proceed to the next task.

        Logs the next task request and quits the application.
        """
        logger.info(f"Next Task=[{self.next_task_name}] requested")
        self.quit()

    def quit(self) -> None:
        """Quit the application."""
        instance = QApplication.instance()
        if instance:
            instance.quit()


if __name__ == "__main__":
    tasks = ["RSVP Calibration", "IntertaskAction", "Matrix Calibration", "IntertaskAction"]

    run_bciui(IntertaskGUI, tasks=tasks, next_task_index=1, exit_callback=lambda: print("Stopping orchestrator"))
