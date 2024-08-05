import os
from typing import Callable
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
)

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
    
    @staticmethod
    def make_list_scroll_area(layout: QLayout) -> QScrollArea:
        scroll_area = QScrollArea()
        items_container = QWidget()
        scroll_area.setWidget(items_container)
        items_container.setLayout(layout)
        scroll_area.setWidgetResizable(True)
        return scroll_area

    # class DynamicList(QVBoxLayout):
    #     """A list of QWidgets that can be dynamically updated"""
    #     widgets: List[QWidget] = []

    #     def __init__(self):
    #         super().__init__()

    #     def add_item(self, item: QWidget):
    #         self.widgets.append(item)
    #         self.addWidget(item)

    #     def remove_item(self, item: QWidget):
    #         self.widgets.remove(item)
    #         self.removeWidget(item)
    #         item.deleteLater()

    #     def clear(self):
    #         for widget in self.widgets:
    #             self.remove_item(widget)
    #         self.widgets = []
        
    #     def list(self):
    #         return self.widgets
    
    # class DynamicItem(QWidget):
    #     def __init__(self, layout: QLayout):
    #         super().__init__()

    #     def remove(self):
    #         self.deleteLater()

    #     def set_parent(parent: DynamicList):
    #         parent.add_item(self)

# --- Experiment registry code ---


def format_experiment_combobox(
    label_text: str, combobox: QComboBox, buttons: Optional[List[QPushButton]]
) -> QHBoxLayout:
    label = QLabel(label_text)
    label.setStyleSheet("font-size: 18px")
    area = QVBoxLayout()
    input_area = QHBoxLayout()
    input_area.setContentsMargins(30, 0, 0, 30)
    area.addWidget(label)
    input_area.addWidget(combobox, 1)
    area.addLayout(input_area)
    return area


def make_field_entry(name: str) -> QWidget:
    layout = QHBoxLayout()
    label = QLabel(name)
    label.setStyleSheet("color: black;")
    layout.addWidget(label)
    remove_button = QPushButton("Remove")
    remove_button.setStyleSheet("background-color: red;")
    remove_button.clicked.connect(lambda: layout.deleteLater())
    layout.addWidget(remove_button)
    widget = QWidget()
    widget.setLayout(layout)
    return widget, remove_button

if __name__ == "__main__":
    app = QApplication([])
    bci = BCIUI("Experiment Registry")
    header = QLabel("Experiment Registry")
    header.setStyleSheet("font-size: 24px")
    # test_label.setStyleSheet("color: red")
    bci.contents.addLayout(BCIUI.centered(header))

    # Add form fields
    form_area = QVBoxLayout()
    form_area.setContentsMargins(30, 30, 30, 0)

    experiment_name_box = format_experiment_combobox("Name", QLineEdit(), None)
    form_area.addLayout(experiment_name_box)

    experiment_summary_box = format_experiment_combobox("Summary", QLineEdit(), None)
    form_area.addLayout(experiment_summary_box)

    experiment_field_box = format_experiment_combobox("Fields", QComboBox(), None)
    form_area.addLayout(experiment_field_box)

    bci.contents.addLayout(form_area)

    create_experiment_button = QPushButton("Create experiment")
   
    bci.contents.addWidget(create_experiment_button)

    fields_scroll_area = QScrollArea()
    fields_items_container = QWidget()

    # fields_items_container.setWidgetResizable(True)
    # fields_scroll_area.setWidget(fields_items_container)
    # fields_scroll_area.setWidgetResizable(True)
    # bci.contents.addWidget(QLabel("Fields"))
    # bci.contents.addWidget(fields_scroll_area)
    fields_content = QVBoxLayout()
    # fields_content = BCIUI.DynamicList()  
    fields_scroll_area = BCIUI.make_list_scroll_area(fields_content)
    label = QLabel("Fields")
    label.setStyleSheet("color: black;")
    bci.contents.addWidget(fields_scroll_area)

    def removeable_field_entry(name: str) -> QWidget:
        widget, button = make_field_entry(name)
        button.clicked.connect(lambda: fields_content.removeWidget(widget))
        button.clicked.connect(lambda: widget.deleteLater())
        return widget

    

    create_experiment_button.clicked.connect(
        lambda: fields_content.addWidget(removeable_field_entry("Field 1"))
        # lambda: fields_content.addWidget(make_field_entry("Field 2"));
    )

    # fields_scroll_area.addWidget(BCIUI.make_list_scroll_area())
    # protocol_scroll_area = QScrollArea()
    # protocol_items_container = QWidget()
    # bci.contents.addWidget(QLabel("Protocol"))
    # bci.contents.addWidget(protocol_scroll_area)

    bci.display()
    # test_label.show()
    app.exec()
