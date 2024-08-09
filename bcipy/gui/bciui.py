import os
from typing import Callable, Type
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
    QMessageBox,
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

    def app(self): ...

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

    def show_alert(self, alert_text: str):
        msg = QMessageBox()
        # This doesn't seem to work at least on mac. TODO: fix this
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(alert_text)
        msg.setWindowTitle("Alert")

        return msg.exec()

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

    def list_property(self, prop: str):
        return [widget.data[prop] for widget in self.widgets]


# --- Experiment registry code ---
from bcipy.helpers.load import load_fields, load_experiments
from bcipy.helpers.save import save_experiment_data
from bcipy.config import DEFAULT_ENCODING, DEFAULT_EXPERIMENT_PATH, DEFAULT_FIELD_PATH, EXPERIMENT_FILENAME, FIELD_FILENAME
import json


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

    def load_fields(path: str = f'{DEFAULT_FIELD_PATH}/{FIELD_FILENAME}') -> dict:
        """Load Fields.

        PARAMETERS
        ----------
        :param: path: string path to the fields file.

        Returns
        -------
            A dictionary of fields, with the following format:
                {
                    "field_name": {
                        "help_text": "",
                        "type": ""
                }

        """
        with open(path, 'r', encoding=DEFAULT_ENCODING) as json_file:
            return json.load(json_file)
    
    def create_experiment(self):
        existing_experiments = load_experiments()
        experiment_name = self.experiment_name_input.text()
        if not experiment_name:
            self.show_alert("Please specify an experiment name")
            return
        experiment_summary = self.experiment_summary_input.text()
        if not experiment_summary:
            self.show_alert("Please specify an experiment summary")
            return
        fields = self.fields_content.list()

        field_list = [
            {
                field["field_name"]: {
                    "anonymize": field["anonymous"],
                    "required": not field["optional"],
                }
            }
            for field in fields
        ]

        existing_experiments[experiment_name] = {
            "fields": field_list,
            "summary": experiment_summary,
        }
        save_experiment_data(existing_experiments, load_fields(), DEFAULT_EXPERIMENT_PATH, EXPERIMENT_FILENAME)
        self.show_alert("created experiment")

    def app(self):
        # Add form fields
        header = QLabel("Experiment Registry")
        header.setStyleSheet("font-size: 24px")
        self.contents.addLayout(BCIUI.centered(header))
        form_area = QVBoxLayout()
        form_area.setContentsMargins(30, 30, 30, 0)

        self.experiment_name_input = QLineEdit()
        experiment_name_box = self.format_experiment_combobox(
            "Name", self.experiment_name_input, None
        )
        form_area.addLayout(experiment_name_box)

        self.experiment_summary_input = QLineEdit()
        experiment_summary_box = self.format_experiment_combobox(
            "Summary", self.experiment_summary_input, None
        )
        form_area.addLayout(experiment_summary_box)

        self.field_input = QComboBox()
        self.field_input.addItems(load_fields())
        add_field_button = QPushButton("Add")
        new_field_button = QPushButton("New")
        form_area.addLayout(
            self.format_experiment_combobox(
                "Fields", self.field_input, [add_field_button, new_field_button]
            )
        )

        self.contents.addLayout(form_area)

        fields_scroll_area = QScrollArea()

        self.fields_content = DynamicList()
        fields_scroll_area = BCIUI.make_list_scroll_area(self.fields_content)
        label = QLabel("Fields")
        label.setStyleSheet("color: black;")
        self.contents.addWidget(fields_scroll_area)

        def add_field():
            if self.field_input.currentText() in self.fields_content.list_property(
                "field_name"
            ):
                self.show_alert("Field already added")
                return
            self.fields_content.add_item(
                self.make_field_entry(self.field_input.currentText())
            )

        add_field_button.clicked.connect(add_field)
        create_experiment_button = QPushButton("Create experiment")
        create_experiment_button.clicked.connect(self.create_experiment)
        self.contents.addWidget(create_experiment_button)

def run_bciui(ui: Type[BCIUI]):
    app = QApplication([])
    ui_instance = ui()
    ui_instance.display()
    app.exec()

if __name__ == "__main__":
    run_bciui(ExperimentRegistry)
