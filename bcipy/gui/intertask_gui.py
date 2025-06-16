import logging
from typing import Callable, List

from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QProgressBar,
                             QPushButton)

from bcipy.config import SESSION_LOG_FILENAME
from bcipy.gui.bciui import BCIUI, run_bciui

logger = logging.getLogger(SESSION_LOG_FILENAME)


class IntertaskGUI(BCIUI):

    action_name = "IntertaskAction"

    def __init__(
        self,
        next_task_index: int,
        tasks: List[str],
        exit_callback: Callable,
    ):
        self.tasks = tasks
        self.current_task_index = next_task_index
        self.next_task_name = tasks[self.current_task_index]
        self.total_tasks = len(tasks)
        self.task_progress = next_task_index
        self.callback = exit_callback
        super().__init__("Progress", 800, 150)
        self.setProperty("class", "inter-task")

    def app(self):
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

    def stop_tasks(self):
        # This should exit Task executions
        logger.info(f"Stopping Tasks... user requested. Using callback: {self.callback}")
        self.callback()
        self.quit()
        logger.info("Tasks Stopped")

    def next(self):
        logger.info(f"Next Task=[{self.next_task_name}] requested")
        self.quit()

    def quit(self):
        QApplication.instance().quit()


if __name__ == "__main__":
    tasks = ["RSVP Calibration", "IntertaskAction", "Matrix Calibration", "IntertaskAction"]

    run_bciui(IntertaskGUI, tasks=tasks, next_task_index=1, exit_callback=lambda: print("Stopping orchestrator"))
