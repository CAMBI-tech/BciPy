from typing import List, Optional
from PyQt6.QtWidgets import (
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
)
from bcipy.gui.bciui import BCIUI, DynamicItem, DynamicList, SmallButton, run_bciui
from bcipy.helpers.load import load_fields, load_experiments
from bcipy.helpers.save import save_experiment_data
from bcipy.config import (
    DEFAULT_ENCODING,
    DEFAULT_EXPERIMENT_PATH,
    DEFAULT_FIELD_PATH,
    EXPERIMENT_FILENAME,
    FIELD_FILENAME,
    BCIPY_ROOT,
)
from bcipy.task.registry import TaskRegistry
import subprocess
from bcipy.task.orchestrator.protocol import serialize_protocol
import json


class ExperimentRegistry(BCIUI):

    task_registry: TaskRegistry

    def __init__(self):
        super().__init__("Experiment Registry", 600, 700)
        self.task_registry = TaskRegistry()

    def format_experiment_combobox(
        self, label_text: str, combobox: QComboBox, buttons: Optional[List[QPushButton]]
    ) -> QVBoxLayout:
        """
        Create a formatted widget for a the experiment comboboxes with optional buttons.

        PARAMETERS
        ----------
        :param: label_text: string text to display above the combobox.
        :param: combobox: the combobox.
        :param: buttons: list of buttons to add to right side of the combobox.

        Returns
        -------
            A QVBoxLayout with the label, combobox, and buttons.
        """
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 18px")
        area = QVBoxLayout()
        input_area = QHBoxLayout()
        input_area.setContentsMargins(15, 0, 0, 15)
        area.addWidget(label)
        input_area.addWidget(combobox, 1)
        if buttons:
            for button in buttons:
                input_area.addWidget(button)
        area.addLayout(input_area)
        return area

    def make_task_entry(self, name: str) -> DynamicItem:
        """
        Create a formatted widget for a task entry.

        PARAMETERS
        ----------
        :param: name: string name of the task that will be displayed.

        Returns
        -------
            A DynamicItem widget with the 'task_name' property set as it's data.
        """
        layout = QHBoxLayout()
        label = QLabel(name)
        label.setStyleSheet("color: black;")
        layout.addWidget(label)
        widget = DynamicItem()

        # The indices will have to be updated to reflect the actual index we want to move to
        move_up_button = SmallButton(" ▲ ")
        move_up_button.clicked.connect(
            lambda: self.protocol_contents.move_item(
                widget, max(self.protocol_contents.index(widget) - 1, 0)
            )
        )
        move_down_button = SmallButton(" ▼ ")
        move_down_button.clicked.connect(
            lambda: self.protocol_contents.move_item(
                widget,
                min(
                    self.protocol_contents.index(widget) + 1,
                    len(self.protocol_contents) - 1,
                ),
            )
        )
        layout.addWidget(move_up_button)
        layout.addWidget(move_down_button)

        remove_button = SmallButton("Remove")
        remove_button.setStyleSheet("background-color: red;")
        remove_button.clicked.connect(
            lambda: layout.deleteLater()
        )  # This may not be needed
        remove_button.clicked.connect(lambda: widget.remove())
        layout.addWidget(remove_button)

        widget.data = {"task_name": name}
        widget.setLayout(layout)
        return widget

    def make_field_entry(self, name: str) -> DynamicItem:
        """
        Create a formatted widget for a field entry.

        PARAMETERS
        ----------
        :param: name: string name of the field that will be displayed.

        Returns
        -------
            A DynamicItem widget with the 'field_name', 'anonymous', and 'optional' properties set as it's data.
        """
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

    def load_fields(path: str = f"{DEFAULT_FIELD_PATH}/{FIELD_FILENAME}") -> dict:
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
        with open(path, "r", encoding=DEFAULT_ENCODING) as json_file:
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
        task_names = self.protocol_contents.list_property("task_name")
        task_objects = [self.task_registry.get(task_name) for task_name in task_names]
        protocol = serialize_protocol(task_objects)

        existing_experiments[experiment_name] = {
            "fields": field_list,
            "summary": experiment_summary,
            "protocol": protocol,
        }
        save_experiment_data(
            existing_experiments,
            load_fields(),
            DEFAULT_EXPERIMENT_PATH,
            EXPERIMENT_FILENAME,
        )
        self.show_alert("created experiment")

    def create_experiment_field(self) -> None:
        """Create Field.

        Launch to FieldRegistry to create a new field for experiments.
        """
        subprocess.call(
            f"python {BCIPY_ROOT}/gui/experiments/FieldRegistry.py", shell=True
        )

        self.update_field_list()

    def update_field_list(self):
        self.field_input.clear()
        self.field_input.addItems(load_fields())

    def app(self):
        # Add form fields
        self.center_content_vertically = True
        header = QLabel("Experiment Registry")
        header.setStyleSheet("font-size: 24px")
        self.contents.addLayout(BCIUI.centered(header))
        form_area = QVBoxLayout()
        form_area.setContentsMargins(30, 0, 30, 0)
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

        def add_field():
            if self.field_input.currentText() in self.fields_content.list_property(
                "field_name"
            ):
                self.show_alert("Field already added")
                return
            self.fields_content.add_item(
                self.make_field_entry(self.field_input.currentText())
            )

        def add_task():
            self.protocol_contents.add_item(
                self.make_task_entry(self.experiment_protocol_input.currentText())
            )

        self.experiment_protocol_input = QComboBox()
        self.experiment_protocol_input.addItems(self.task_registry.list())
        add_task_button = QPushButton("Add")
        add_task_button.clicked.connect(add_task)
        experiment_protocol_box = self.format_experiment_combobox(
            "Protocol", self.experiment_protocol_input, [add_task_button]
        )
        form_area.addLayout(experiment_protocol_box)

        self.field_input = QComboBox()
        self.field_input.addItems(load_fields())
        add_field_button = QPushButton("Add")
        new_field_button = QPushButton("New")
        new_field_button.clicked.connect(self.create_experiment_field)
        form_area.addLayout(
            self.format_experiment_combobox(
                "Fields", self.field_input, [add_field_button, new_field_button]
            )
        )

        self.contents.addLayout(form_area)

        scroll_area_layout = QHBoxLayout()

        self.fields_content = DynamicList()
        fields_scroll_area = BCIUI.make_list_scroll_area(self.fields_content)
        label = QLabel("Fields")
        label.setStyleSheet("color: black;")
        scroll_area_layout.addWidget(fields_scroll_area)

        protocol_scroll_area = QScrollArea()
        self.protocol_contents = DynamicList()
        protocol_scroll_area = BCIUI.make_list_scroll_area(self.protocol_contents)
        label = QLabel("Protocol")
        label.setStyleSheet("color: black;")
        scroll_area_layout.addWidget(protocol_scroll_area)

        self.contents.addLayout(scroll_area_layout)

        add_field_button.clicked.connect(add_field)
        create_experiment_button = QPushButton("Create experiment")
        create_experiment_button.clicked.connect(self.create_experiment)
        self.contents.addWidget(create_experiment_button)


if __name__ == "__main__":
    run_bciui(ExperimentRegistry)
