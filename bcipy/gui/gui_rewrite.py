import os
from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QPushButton


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
    label = QLabel("Name")
    label.setStyleSheet("font-size: 18px")
    form_area.addWidget(label)
    bci.contents.addLayout(form_area)

    test_button = QPushButton("Test")
    bci.contents.addWidget(test_button)


    bci.display()
    # test_label.show()
    app.exec()