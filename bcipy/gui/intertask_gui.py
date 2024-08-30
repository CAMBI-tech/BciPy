from bcipy.gui.bciui import BCIUI, run_bciui
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QProgressBar,
)


class IntertaskGUI(BCIUI):

    def __init__(
        self, next_task_name: str, total_tasks: int = 0, current_task_index: int = 0
    ):
        self.total_tasks = total_tasks
        self.current_task_index = current_task_index
        self.next_task_name = next_task_name
        super().__init__("Progress", 400, 100)

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
        # This should be replaced with a method that stops orchestrator execution
        self.stop_button.clicked.connect(QApplication.instance().quit)


if __name__ == "__main__":
    # test values
    run_bciui(IntertaskGUI, 'Placeholder Task Name', 3, 2)
