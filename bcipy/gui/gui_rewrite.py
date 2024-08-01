import os
from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QComboBox

from typing import Optional, List

class BCIUI(QWidget):
    contents: QVBoxLayout
    def __init__(self, title="BCIUI", default_width=500, default_height=600):
        super().__init__()
        self.resize(default_width, default_height)
        self.setWindowTitle(title)
        self.contents = QVBoxLayout()
        self.setLayout(self.contents)
    
    def apply_stylesheet(self):
        gui_dir = os.path.dirname(os.path.realpath(__file__))
        stylesheet_path = os.path.join(gui_dir, "bcipy_stylesheet.css")
        with open(stylesheet_path, "r") as f:
            stylesheet = f.read()
        self.setStyleSheet(stylesheet)
        
    def display(self):
        # Push contents to the top of the window
        self.contents.addStretch()
        self.apply_stylesheet()
        self.show()

    @staticmethod
    def centered(widget: QWidget):
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(widget)
        layout.addStretch()
        return layout

# --- Experiment registry code ---

def format_experiment_combobox(label_text: str, combobox: QComboBox, buttons: Optional[List[QPushButton]]) -> QHBoxLayout:
    label = QLabel(label_text)
    label.setStyleSheet("font-size: 18px")
    area = QVBoxLayout()
    input_area = QHBoxLayout()
    input_area.setContentsMargins(30, 0, 0, 30)
    area.addWidget(label)
    input_area.addWidget(combobox, 1)
    area.addLayout(input_area)
    return area

if __name__ == '__main__':
    app = QApplication([])
    bci = BCIUI()
    header = QLabel("Experiment Registry")
    header.setStyleSheet("font-size: 24px")
    # test_label.setStyleSheet("color: red")
    bci.contents.addLayout(BCIUI.centered(header))

    # Add form fields 
    form_area = QVBoxLayout()
    form_area.setContentsMargins(30, 50, 30, 0)

    experiment_name_box = format_experiment_combobox("Name", QComboBox(), None)
    form_area.addLayout(experiment_name_box)

    experiment_summary_box = format_experiment_combobox("Summary", QComboBox(), None)
    form_area.addLayout(experiment_summary_box)

    experiment_field_box = format_experiment_combobox("Fields", QComboBox(), None)
    form_area.addLayout(experiment_field_box)

    

    bci.contents.addLayout(form_area)

    create_experiment_button = QPushButton("Create experiment")
    bci.contents.addWidget(create_experiment_button)


    bci.display()
    # test_label.show()
    app.exec()