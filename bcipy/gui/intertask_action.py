from bcipy.gui.bciui import BCIUI, run_bciui
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import (
    QWidget,
    QApplication,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QScrollArea,
    QLineEdit,
    QLayout,
    QSizePolicy,
    QMessageBox,
    QProgressBar,
)


class IntertaskAction(BCIUI):

    def __init__(self, total_tasks: int = 0, current_task_index: int = 0):
        self.total_tasks = total_tasks
        self.current_task_index = current_task_index
        super().__init__("Progress", 400, 300)

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

        self.contents.addStretch(1)
        self.next_button = QPushButton("Next")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("background-color: red")
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.next_button)
        self.contents.addLayout(buttons_layout)


if __name__ == "__main__":
    # test values
    run_bciui(IntertaskAction, 3, 2)
