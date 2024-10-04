from typing import Callable

from PyQt6.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QPushButton,
    QProgressBar,
)
from bcipy.gui.bciui import BCIUI, run_bciui



class IntertaskGUI(BCIUI):

    def __init__(
        self,
        next_task_name: str,
        current_task_index: int,
        total_tasks: int,
        exit_callback: Callable,
    ):
        self.total_tasks = total_tasks
        self.current_task_index = current_task_index
        self.next_task_name = next_task_name
        self.callback = exit_callback
        super().__init__("Progress", 800, 150)

    def app(self):
        self.contents.addLayout(BCIUI.centered(QLabel("Experiment Progress")))

        progress_container = QHBoxLayout()
        progress_container.addWidget(
            QLabel(f"({self.current_task_index}/{self.total_tasks})")
        )
        self.progress = QProgressBar()
        self.progress.setValue(int(self.current_task_index / self.total_tasks * 100))
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

        self.next_button.clicked.connect(self.close)
        self.stop_button.clicked.connect(self.stop_orchestrator)

    def stop_orchestrator(self):
        # This should stop the orchestrator execution
        self.callback()
        self.close()


if __name__ == "__main__":
    # test values
    run_bciui(IntertaskGUI, "Placeholder Task Name", 1, 3, lambda: print("Stopping orchestrator"))
