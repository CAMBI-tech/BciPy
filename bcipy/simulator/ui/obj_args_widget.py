from typing import Any, Callable, List, Optional, Tuple, Type

# pylint: disable=E0611
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFormLayout, QLineEdit, QWidget

from bcipy.core.list import find_index
from bcipy.gui.parameters.params_form import clear_layout
from bcipy.simulator.ui.gui_utils import InputField, get_inputs


class ObjectArgInputs(QWidget):
    """Widget with inputs for parameters needed to instantiate an object for a
    given class.
    """

    def __init__(self,
                 parent: Optional[QWidget] = None,
                 change_event: Optional[Callable] = None,
                 object_type: Optional[Type[Any]] = None):
        super().__init__(parent=parent)
        self.change_event = change_event
        self.form_layout = QFormLayout()
        self.form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft
                                          | Qt.AlignmentFlag.AlignVCenter)
        self.form_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.object_type = object_type
        self.input_fields = self._init_input_fields()
        self.controls = self._create_inputs()
        self._add_controls()
        self.setLayout(self.form_layout)

    def set_object_type(self, object_type: Optional[Type[Any]]) -> None:
        """Set the class for which we should provide inputs."""
        self.object_type = object_type
        self.input_fields = self._init_input_fields()
        self.controls = self._create_inputs()
        self._add_controls()

    def required_inputs_provided(self) -> bool:
        """Check if required inputs have been filled out."""
        for label, control in self.controls:
            field = self._field_definition(label)
            if field.required and not self._input_value(field, control):
                return False
        return True

    def value(self) -> str:
        """Returns a json string representing a dict of name: value for each input."""
        vals = ", ".join([
            self._json_name_value(name, control)
            for (name, control) in self.controls
        ])
        return f"{{{vals}}}"

    def _init_input_fields(self) -> List[InputField]:
        """Determine the input fields for the configured object type."""
        if self.object_type:
            return get_inputs(self.object_type)
        return []

    def _field_definition(self, name: str) -> InputField:
        """Get the InputField for the control with the given name."""
        index = find_index(self.input_fields, name, key=lambda item: item.name)
        if index is None:
            raise KeyError(f"Not found: {name}")
        return self.input_fields[index]

    def _json_name_value(self, name: str, control: QWidget) -> str:
        """Returns a json partial for a name: value, quoting the value according
        to the input_type
        """
        field = self._field_definition(name)
        value = self._input_value(field, control)

        if not value:
            return f'"{name}": null'
        if field.input_type == 'str':
            value = f'"{value}"'

        return f'"{name}": {value}'

    def _create_inputs(self) -> List[Tuple[str, QWidget]]:
        """Create inputs for the inputs for associated args."""
        return [(input_field.name, self._create_input(input_field))
                for input_field in self.input_fields]

    def _create_input(self, input_field: InputField) -> QWidget:
        """Create an input for the given InputField"""
        # currently only supports int, str, and float. Text inputs are used
        # since the spinbox inputs do not allow a None value and default to
        # 0, which is not always correct.
        input_map = {'str': QLineEdit, 'int': QLineEdit, 'float': QLineEdit}

        make_control = input_map[input_field.input_type]
        control = make_control()
        if input_field.value is not None:
            control.setText(str(input_field.value))
        control.textChanged.connect(self.change)
        return control

    def _input_value(self, field: InputField, control: QWidget) -> Any:
        """Get the cast value for the provided input."""
        # TODO: cast this according to its type
        return control.text()

    def _add_controls(self) -> None:
        """Add each input to the layout."""
        clear_layout(self.form_layout)
        for (label, control) in self.controls:
            self.form_layout.addRow(label, control)

    def change(self):
        """Called when the sampler or any inputs change."""
        if self.change_event:
            self.change_event()
