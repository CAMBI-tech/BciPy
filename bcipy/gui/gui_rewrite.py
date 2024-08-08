import os
from typing import Callable
from PyQt6.QtCore import pyqtSignal
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

    def app(self):
        ...

    def apply_stylesheet(self):
        gui_dir = os.path.dirname(os.path.realpath(__file__))
        stylesheet_path = os.path.join(gui_dir, "bcipy_stylesheet.css")
        with open(stylesheet_path, "r") as f:
            stylesheet = f.read()
        self.setStyleSheet(stylesheet)

    def display(self):
        # Push contents to the top of the window
        # self.contents.addStretch()
        self.app()
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
    def make_list_scroll_area(widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)
        return scroll_area

    @staticmethod
    def make_toggle(
        on_button: QPushButton,
        off_button: QPushButton,
        on_action: Optional[Callable] = lambda: None,
        off_action: Optional[Callable] = lambda: None,
    ):
        """Connects two buttons to toggle between eachother and call passed methods"""
        off_button.hide()

        def toggle_off():
            on_button.hide()
            off_button.show()
            off_action()

        def toggle_on():
            on_button.show()
            off_button.hide()
            on_action()

        on_button.clicked.connect(toggle_off)
        off_button.clicked.connect(toggle_on)


class SmallButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setProperty("class", "small-button")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


class DynamicItem(QWidget):
    """A widget that can be dynamically added and removed from the ui"""

    on_remove: pyqtSignal = pyqtSignal()
    data: dict = {}

    def remove(self):
        """Remove the widget from it's parent DynamicList, removing it from the UI and deleting it"""
        self.on_remove.emit()


class DynamicList(QWidget):
    """A list of QWidgets that can be dynamically updated"""

    widgets: List[QWidget] = []

    def __init__(self, layout: Optional[QLayout] = QVBoxLayout()):
        super().__init__()
        self.setLayout(layout)

    def add_item(self, item: DynamicItem):
        self.widgets.append(item)
        item.on_remove.connect(lambda: self.remove_item(item))
        self.layout().addWidget(item)

    def remove_item(self, item: DynamicItem):
        self.widgets.remove(item)
        self.layout().removeWidget(item)
        item.deleteLater()

    def clear(self):
        for widget in self.widgets:
            self.layout().removeWidget(widget)
            widget.deleteLater()
        self.widgets = []

    def list(self):
        return [widget.data for widget in self.widgets]


# --- Experiment registry code ---
from bcipy.helpers.load import load_fields, load_experiments
from bcipy.helpers.save import save_experiment_data
from bcipy.config import DEFAULT_EXPERIMENT_PATH, EXPERIMENT_FILENAME



class ExperimentRegistry(BCIUI):
    
    def format_experiment_combobox(
        self, label_text: str, combobox: QComboBox, buttons: Optional[List[QPushButton]]
    ) -> QHBoxLayout:
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 18px")
        area = QVBoxLayout()
        input_area = QHBoxLayout()
        input_area.setContentsMargins(30, 0, 0, 30)
        area.addWidget(label)
        input_area.addWidget(combobox, 1)
        if buttons:
            for button in buttons:
                input_area.addWidget(button)
        area.addLayout(input_area)
        return area

    def make_field_entry(self, name: str) -> QWidget:
        layout = QHBoxLayout()
        label = QLabel(name)
        label.setStyleSheet("color: black;")
        layout.addWidget(label)
        widget = DynamicItem()

        remove_button = SmallButton("Remove")
        remove_button.setStyleSheet("background-color: red;")
        remove_button.clicked.connect(lambda: layout.deleteLater())

        anonymous_button = SmallButton("Anonymous")
        onymous_button = SmallButton("Onymous")
        # anonymous_button.clicked.connect(lambda: widget.data.update({"anonymous": True}))
        # onymous_button.clicked.connect(lambda: widget.data.update({"anonymous": False}))
        BCIUI.make_toggle(
            anonymous_button,
            onymous_button,
            on_action=lambda: widget.data.update({"anonymous": True}),
            off_action=lambda: widget.data.update({"anonymous": False}),
        )
        layout.addWidget(anonymous_button)
        layout.addWidget(onymous_button)
        anonymous_button.setStyleSheet("background-color: black;")

        optional_button = SmallButton("Optional")
        required_button = SmallButton("Required")
        BCIUI.make_toggle(
            optional_button,
            required_button,
            on_action=lambda: widget.data.update({"optional": True}),
            off_action=lambda: widget.data.update({"optional": False}),
        )
        layout.addWidget(optional_button)
        layout.addWidget(required_button)

        layout.addWidget(remove_button)
        widget.data = {"field_name": name, "anonymous": True, "optional": True}
        remove_button.clicked.connect(lambda: widget.remove())
        widget.setLayout(layout)
        return widget

    def app(self):
        # Add form fields
        header = QLabel("Experiment Registry")
        header.setStyleSheet("font-size: 24px")
        self.contents.addLayout(BCIUI.centered(header))
        form_area = QVBoxLayout()
        form_area.setContentsMargins(30, 30, 30, 0)

        experiment_name_box = self.format_experiment_combobox("Name", QLineEdit(), None)
        form_area.addLayout(experiment_name_box)

        experiment_summary_box = self.format_experiment_combobox("Summary", QLineEdit(), None)
        form_area.addLayout(experiment_summary_box)

        field_input = QComboBox()
        field_input.addItems(load_fields())
        add_field_button = QPushButton("Add")
        new_field_button = QPushButton("New")
        form_area.addLayout(
            self.format_experiment_combobox(
                "Fields", field_input, [add_field_button, new_field_button]
            )
        )

        bci.contents.addLayout(form_area)

        fields_scroll_area = QScrollArea()
        fields_items_container = QWidget()

        fields_content = DynamicList()
        fields_scroll_area = BCIUI.make_list_scroll_area(fields_content)
        label = QLabel("Fields")
        label.setStyleSheet("color: black;")
        bci.contents.addWidget(fields_scroll_area)
        add_field_button.clicked.connect(
            lambda: fields_content.add_item(self.make_field_entry(field_input.currentText()))
        )

        create_experiment_button = QPushButton("Create experiment")
        create_experiment_button.clicked.connect(lambda: print('save here'))
        bci.contents.addWidget(create_experiment_button)


if __name__ == "__main__":
    app = QApplication([])
    bci = ExperimentRegistry()
    bci.display()
    app.exec()
